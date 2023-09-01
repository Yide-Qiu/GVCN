import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import pdb


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.dropout0(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h

class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        # z_dim,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        norm="none",
    ):
        super(GATConv, self).__init__()
        if norm not in ("none", "both"):
            raise DGLError('Invalid norm value. Must be either "none", "both".' ' But got "{}".'.format(norm))
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats) # 返回元组(infeats, infeats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        # self._z_dim = z_dim
        if isinstance(in_feats, tuple): # 默认false
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.mu_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.mu_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.lam_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.lam_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.mu_src, gain=gain/30)
        nn.init.xavier_normal_(self.mu_dst, gain=gain/30)
        nn.init.xavier_normal_(self.lam_src, gain=gain/30)
        nn.init.xavier_normal_(self.lam_dst, gain=gain/30)
        # nn.init.xavier_normal_(self.attn_l, gain=gain)
        # nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def reparameterize(self, mu, logvar) :
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, graph, feat):
        with graph.local_scope(): # 相当于制作graph的副本，修改graph不影响其原始值
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                # FC
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                # FC
                h_src = h_dst = self.feat_drop(feat) # !
                feat_src, feat_dst = h_src, h_dst
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]

            if self._norm == "both":  # 默认设置有这个
                degs = graph.out_degrees().float().clamp(min=1) # 度为0的加自环这和外面的直接全部加自环不太匹配
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1) # 根据feat维度增加norm维度，增加数量为feat维度减一
                norm = torch.reshape(norm, shp) # [num_n] -> [num_n , 1] / [num_n , 1, 1]
                feat_src = feat_src * norm # 做D^(-.5)*X  个人推测他在模拟DADX  先做DX  再AX(MPNN)

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            # pdb.set_trace()

            mu_src = (feat_src * self.mu_src).sum(dim=-1).unsqueeze(-1)
            lam_src = (feat_src * self.lam_src).sum(dim=-1).unsqueeze(-1)
            mu_dst = (feat_dst * self.mu_dst).sum(dim=-1).unsqueeze(-1)
            lam_dst = (feat_dst * self.lam_src).sum(dim=-1).unsqueeze(-1)

            z_src = self.reparameterize(mu_src, lam_src)
            z_dst = self.reparameterize(mu_dst, lam_dst)

            # vae 用ev eu 替代el er    ppmi 做ev eu参数 或 el er参数
            # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            # graph.srcdata.update({"ft": feat_src, "el": el})
            # graph.dstdata.update({"er": er})
            ppmi = graph.edata["ppmi"].view(-1, 1, 1)
            graph.srcdata.update({"ft": feat_src, "el": z_src})
            graph.dstdata.update({"er": z_dst})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e")) # 对节点做不同的两个映射，以边为单位将映射结果加起来作为边权
            # 则VAE可以对全部节点变分成一个值(低维隐向量也行)，然后以边为单位把节点对加起来，两种映射可能有嵌入空间偏差的风险吧
            # 如果要尽量做成和GAT一致的话，我们可以VAE两次，取不同的值
            e = self.leaky_relu(graph.edata.pop("e")) # 取e并做lre
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
                
            e2 = torch.zeros_like(e)
            e2[eids]=e[eids]
            
            # pdb.set_trace()
            e2 = e2 * torch.log(torch.log(ppmi).clamp(min=1)).clamp(min=1)
            # compute softmax
            # pdb.set_trace()
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e2)) # local softmax
            # message passing
            # 以边为单位 用ume制作message “m”=src’“ft”*e’a 用sum聚合 ft=sum(ft'N(e’“m”)) 
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._norm == "both":
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            # return rst
            return rst, mu_src+mu_dst, lam_src+lam_dst


class VAE(nn.Module):
    def __init__(
        self, in_feats, n_classes, n_hidden, n_layers, n_heads, activation, 
        dropout=0.0, input_drop=0.0, attn_drop=0.0, edge_drop=0.0, norm="none"
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.biases = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            # in_channels = n_heads if i > 0 else 1
            out_channels = n_heads

            self.convs.append(GATConv(in_hidden, out_hidden, num_heads=n_heads, attn_drop=attn_drop, edge_drop=edge_drop, norm=norm))

            self.linear.append(nn.Linear(in_hidden, out_channels * out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = Bias(n_classes)

        self.input_dropout = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_dropout(h)
        h_last = h
        for i in range(self.n_layers):
            conv, mu, log_var = self.convs[i](graph, h)
            linear = self.linear[i](h).view(conv.shape)
            # pdb.set_trace()
            h = conv + linear

            if i < self.n_layers - 1:
                h = h.flatten(1)
                if h_last.shape[-1] == h.shape[-1]:
                    h = h + h_last
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
                h_last = h

        h = h.mean(1)
        h = self.bias_last(h)

        return h, mu, log_var
