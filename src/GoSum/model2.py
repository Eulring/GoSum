import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self,embed_size, gtype):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(embed_size, embed_size, bias=False)
        self.attn_fc = nn.Linear(2 * embed_size, 1, bias=False)
        self.ffn = PositionwiseFeedForward(embed_size, 512, 0.1)
        self.gtype = gtype
    
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, c, s):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        cnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 2)
        onode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 3)
        gnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 4)

        # ipdb.set_trace()
        if self.gtype == 's2c':
            scedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 1) & (edges.dst["unit"] == 2))
            zs = self.fc(s)
            # print(len(snode_id))
            # print(s.shape)
            g.nodes[snode_id].data['z'] = zs
            g.apply_edges(self.edge_attention, edges=scedge_id)
            g.pull(cnode_id, self.message_func, self.reduce_func)
            g.ndata.pop('z')
            h = g.ndata.pop('sh')
            hc = h[cnode_id]
            hc = F.elu(hc) + c
            hc = self.ffn(hc.unsqueeze(0)).squeeze(0)
            # ipdb.set_trace()
            return hc
        
        if self.gtype == 'c2s':
            csedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 1))
            zc = self.fc(c)
            g.nodes[cnode_id].data['z'] = zc
            g.apply_edges(self.edge_attention, edges=csedge_id)
            g.pull(snode_id, self.message_func, self.reduce_func)
            g.ndata.pop('z')
            h = g.ndata.pop('sh')
            hs = h[snode_id]
            hs = F.elu(hs) + s
            hs = self.ffn(hs.unsqueeze(0)).squeeze(0)
            # ipdb.set_trace()
            return hs

        if self.gtype == 'c2g':
            cgedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 4))
            zc = self.fc(c)
            g.nodes[cnode_id].data['z'] = zc
            g.apply_edges(self.edge_attention, edges=cgedge_id)
            g.pull(gnode_id, self.message_func, self.reduce_func)
            g.ndata.pop('z')
            h = g.ndata.pop('sh')
            hg = h[gnode_id]
            hg = F.elu(hg) + s
            hg = self.ffn(hg.unsqueeze(0)).squeeze(0)
            # ipdb.set_trace()
            return hg

        if self.gtype == 'c2c':
            ccedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == 2) & (edges.dst["unit"] == 2))
            zc = self.fc(c)
            g.nodes[cnode_id].data['z'] = zc
            g.apply_edges(self.edge_attention, edges=ccedge_id)
            g.pull(gnode_id, self.message_func, self.reduce_func)
            g.ndata.pop('z')
            h = g.ndata.pop('sh')
            hc = h[cnode_id]
            hc = F.elu(hc) + c
            hc = self.ffn(hc.unsqueeze(0)).squeeze(0)
            # ipdb.set_trace()
            return hc


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output



class SGraph(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.embed_size = emb_size
        self._n_iter = 1
        self.GATS2C = GATLayer(emb_size, 's2c')
        self.GATC2S = GATLayer(emb_size, 'c2s')
        # self.GATC2G = GATLayer(emb_size, 'c2g')
        self.GATC2C = GATLayer(emb_size, 'c2c')

    def forward(self, graph, sent_embed, sec_embed, gsent_embed):

        # Gnn module
        sec_state = self.GATS2C(graph, sec_embed, sent_embed)
        sec_state_ = self.GATC2C(graph, sec_state, sec_state)

        sent_state = self.GATC2S(graph, sec_state_, sent_embed)

        # sec_state_ = self.GATC2C(graph, sec_state, sec_state)
        # gsent_state = self.GATC2G(graph, sec_state, gsent_embed)

        return sent_state, sec_state_, sent_state