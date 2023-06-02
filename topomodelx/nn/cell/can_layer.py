"""HodgeNet."""
import torch
from torch import nn

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class CANLayer(torch.nn.Module):
    """Layer of a Convolutional Cell Complex Network (CCXN).

    Implementation of a simplified version of the CCXN layer proposed in [HIZ20]_.

    This layer is composed of two convolutional layers:
    1. A convolutional layer sending messages from nodes to nodes.
    2. A convolutional layer sending messages from edges to faces.
    Optionally, attention mechanisms can be used.

    Notes
    -----
    This is the architecture proposed for entire complex classification.

    References
    ----------
    .. [HIZ20] Hajij, Istvan, Zamzmi. Cell Complex Neural Networks.
        Topological Data Analysis and Beyond Workshop at NeurIPS 2020.
        https://arxiv.org/pdf/2010.00743.pdf

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).
    att : bool
        Whether to use attention.
    """

    def __init__(self, channels, activation="sigmoid"):
        super().__init__()
        self.conv = Conv(in_channels=channels, out_channels=channels)
        self.linear = nn.Linear(channels, channels, bias=False)
        self.aggr = Aggregation(update_func=activation)

    def forward(self, x, down_laplacian, up_laplacian):
        r"""Forward pass.

        The forward pass was initially proposed in [HIZ20]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        The forward pass of this layer is composed of two steps.

        1. The convolution from nodes to nodes is given by an adjacency message passing scheme (AMPS):

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(0 \rightarrow 1 \rightarrow 0)}
                = M_{\mathcal{L}_\uparrow}(h_x^{(0)}, h_y^{(0)}, \Theta^{(y \rightarrow x)})\\
            &🟧 \quad m_x^{(0 \rightarrow 1 \rightarrow 0)}
                = \text{AGG}_{y \in \mathcal{L}_\uparrow(x)}(m_{y \rightarrow \{z\} \rightarrow x}^{0 \rightarrow 1 \rightarrow 0})\\
            &🟩 \quad m_x^{(0)}
                = m_x^{(0 \rightarrow 1 \rightarrow 0)}\\
            &🟦 \quad h_x^{t+1,(0)}
                = U^{t}(h_x^{(0)}, m_x^{(0)})
            \end{align*}

        2. The convolution from edges to faces is given by cohomology message passing scheme, using the coboundary neighborhood:

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow x}^{(r' \rightarrow r)}
                = M^t_{\mathcal{C}}(h_{x}^{t,(r)}, h_y^{t,(r')}, x, y)\\
            &🟧 \quad m_x^{(r' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(r' \rightarrow r)}\\
            &🟩 \quad m_x^{(r)}
                = m_x^{(r' \rightarrow r)}\\
            &🟦 \quad h_{x}^{t+1,(r)}
                = U^{t,(r)}(h_{x}^{t,(r)}, m_{x}^{(r)})
            \end{align*}

        References
        ----------
        .. [HIZ20] Hajij, Istvan, Zamzmi. Cell Complex Neural Networks.
            Topological Data Analysis and Beyond Workshop at NeurIPS 2020.
            https://arxiv.org/pdf/2010.00743.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_0_cells, channels]
            Input features on the nodes of the cell complex.
        x_1 : torch.Tensor, shape=[n_1_cells, channels]
            Input features on the edges of the cell complex.
        neighborhood_0_to_0 : torch.sparse
            shape=[n_0_cells, n_0_cells]
            Neighborhood matrix mapping nodes to nodes (A_0_up).
        neighborhood_1_to_2 : torch.sparse
            shape=[n_2_cells, n_1_cells]
            Neighborhood matrix mapping edges to faces (B_2^T).
        x_2 : torch.Tensor, shape=[n_2_cells, channels]
            Input features on the faces of the cell complex.
            Optional, only required if attention is used between edge s and faces.

        Returns
        -------
        _ : torch.Tensor, shape=[1, num_classes]
            Output prediction on the entire cell complex.
        """
        x_down = self.conv(x, down_laplacian)
        x_up = self.conv(x, up_laplacian)
        x_id = self.linear(x)

        x = self.aggr(x_down, x_up, x_id)
