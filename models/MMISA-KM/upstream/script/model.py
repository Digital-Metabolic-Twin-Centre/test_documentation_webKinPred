import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp,global_mean_pool as gep,global_sort_pool,LayerNorm,TopKPooling
from torch_geometric.utils import dropout_adj
from einops.layers.torch import  Reduce
from functools import reduce



class SKConv(nn.Module):
    def __init__(self, in_channels=320, out_channels=256, stride=1, M=2, r=8, L=32):
        '''
        :param in_channels: 输入通道维度
        :param out_channels: 输出通道维度，输入输出通道维度相同
        :param M: 分支数
        :param r: 特征Z的长度，计算其维度d时所需的比率
        :param L: 默认为32
        采用分组卷积： groups = 32，所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)  # 计算从向量C降维到向量Z的长度d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # 根据分支数量添加不同核的卷积操作
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=32, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)  # 自适应池化到指定维度，这里指定为1，实现GAP
        self.fc1 = nn.Sequential(
            nn.Conv1d(out_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )  # 降维
        self.fc2 = nn.Conv1d(d, out_channels * M, kernel_size=1, bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1使得多个全连接层对应位置进行softmax,保证对应位置a+b+..=1

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            output.append(conv(input))  # [batch_size, out_channels, d]

        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # 逐元素相加生成混合特征U [batch_size, channel, d]
        s = self.global_pool(U)  # [batch_size, channel, 1]
        z = self.fc1(s)  # S->Z降维 [batch_size, d, 1]
        a_b = self.fc2(z)  # Z->a，b升维 [batch_size, out_channels * M, 1]
        a_b = a_b.view(batch_size, self.M, self.out_channels, -1)  # 调整形状 [batch_size, M, out_channels, 1]
        a_b = self.softmax(a_b)  # 使得多个全连接层对应位置进行softmax [batch_size, M, out_channels, 1]

        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # 分割成多个 [batch_size, 1, out_channels, 1], [batch_size, 1, out_channels, 1]
        a_b = list(map(lambda x: x.view(batch_size, self.out_channels, 1), a_b))  # 调整形状 [batch_size, out_channels, 1]
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应不同卷积核输出的U逐元素相乘
        V = reduce(lambda x, y: x + y, V)  # 多个加权后的特征逐元素相加 [batch_size, out_channels, d]
        # V = self.max_pool(V).squeeze(2)
        # V = Reduce('b c t -> b c', 'max')(V)
        # V = V.transpose(1, 2)
        return V  # [batch_size, out_channels, d]









class CNNNET(nn.Module):
    def __init__(self, dim=256, lstm_units=256):
        super(CNNNET, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, lstm_units, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm1d(lstm_units),
            nn.LeakyReLU())

        self.SKConv = SKConv(in_channels=256)


    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in：bs, dim, window out: bs, lstm_units, window
        x = self.SKConv(x)
        return x


class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """

    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        self.W_Q = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.W_K = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.W_V = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(embed_size, embed_size)))
        self.act = nn.ReLU()

        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        attn_w = F.softmax(inner, dim=-1)

        F.dropout(attn_w, p=self.dropout, training=self.training)

        results = torch.matmul(attn_w, Value)

        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))

        results = self.act(results)

        return results





class Selfattention(nn.Module):

    def __init__(self, embed_size, head_num, dropout=0.2):
        super(Selfattention, self).__init__()
        self.vec_wise_net = MultiHeadAttentionInteract(embed_size=embed_size,
                                                       head_num=head_num,
                                                       dropout=dropout)
    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)

        return vec_wise_x
# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, embed_dim=256,n_output=1, num_features_pro=33, num_features_mol=78, dropout=0.2):
        super(GNNNet, self).__init__()

        self.embed_smile = nn.Embedding(65, embed_dim)
        self.embed_prot = nn.Embedding(26, embed_dim)

        self.CNNNET1 = CNNNET(dim=embed_dim, lstm_units=embed_dim)
        self.CNNNET2 = CNNNET(dim=embed_dim, lstm_units=embed_dim)



        self.mol_max_pool = nn.MaxPool1d(100)
        self.pro_max_pool = nn.MaxPool1d(500)

        print('GNNNet Loaded')
        self.n_output = n_output

        self.mol_conv1 = GATConv(num_features_mol, num_features_mol,heads=8)
        self.mol_conv2 = GCNConv(num_features_mol * 8, 256)


        self.mol_fc_g1 = torch.nn.Linear(num_features_mol*4, 512)
        self.mol_fc_g2 = torch.nn.Linear(512, 256)
        # self.mol_fc_g2 = torch.nn.Linear(256, output_dim)


        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GATConv(num_features_pro, num_features_pro,heads=8)
        self.pro_conv2 = GCNConv(num_features_pro * 8, 256)
        # self.pro_conv3 = GATConv(num_features_pro* 8, num_features_pro, heads=8,)

        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro*8, 256)
        self.pro_fc_g2 = torch.nn.Linear(512, 256)
        # self.pro_fc_g2 = torch.nn.Linear(1000, output_dim)




        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        # self.fc1 = nn.Linear(2 * output_dim, 512)
        self.fc1 = nn.Linear(embed_dim*4, embed_dim*2)
        self.out = nn.Linear(embed_dim*2, self.n_output)

        self.feature_interact = Selfattention( embed_size=embed_dim, head_num=8)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, smile,sequence,data_mol,data_pro):



        smile_vectors_onehot = self.embed_smile(smile)
        proteinFeature_onehot = self.embed_prot(sequence)

        mol_pair = self.CNNNET1(smile_vectors_onehot)
        pro_pair = self.CNNNET2(proteinFeature_onehot)
        mol_pair = self.mol_max_pool(mol_pair).squeeze(2)
        pro_pair = self.pro_max_pool(pro_pair).squeeze(2)



        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.leaky_relu(x)

        # x = self.mol_conv3(x, mol_edge_index)
        # x = self.relu(x)

        x = gmp(x, mol_batch)


        #
        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)
        xt = self.dropout(xt)

        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)


        xt = gmp(xt, target_batch)


        all_features = torch.stack([mol_pair,x,pro_pair,xt], dim=2)
        all_features = self.norm(all_features.permute(0, 2, 1))
        all_features = self.feature_interact(all_features)


        # all_features = torch.cat((mol_pair,pro_pair,x), 1)

        xc = self.leaky_relu(self.fc1(all_features))
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
