import torch

MELS_MEAN = torch.FloatTensor(
    [
        -3.9690,
        -4.6535,
        -4.6714,
        -4.4241,
        -4.1878,
        -4.1855,
        -4.4557,
        -4.6717,
        -4.6518,
        -4.5795,
        -4.4660,
        -4.5626,
        -4.6312,
        -4.7365,
        -4.8170,
        -5.0296,
        -5.1707,
        -5.2689,
        -5.3891,
        -5.5151,
        -5.6027,
        -5.6234,
        -5.7600,
        -5.8224,
        -5.8788,
        -5.8744,
        -5.9951,
        -6.0152,
        -6.0584,
        -6.1093,
        -6.1603,
        -6.2054,
        -6.2144,
        -6.2099,
        -6.2672,
        -6.1945,
        -6.2472,
        -6.2191,
        -6.2008,
        -6.2232,
        -6.2474,
        -6.2842,
        -6.3389,
        -6.3736,
        -6.4225,
        -6.4663,
        -6.5121,
        -6.5258,
        -6.5313,
        -6.5100,
        -6.5283,
        -6.5168,
        -6.5587,
        -6.6036,
        -6.6508,
        -6.7079,
        -6.7566,
        -6.8139,
        -6.8673,
        -6.9097,
        -6.9593,
        -7.0071,
        -7.0756,
        -7.1610,
        -7.2428,
        -7.3401,
        -7.4329,
        -7.5403,
        -7.6432,
        -7.7469,
        -7.8342,
        -7.9074,
        -7.9705,
        -8.0207,
        -8.0538,
        -8.0737,
        -8.0811,
        -8.0909,
        -8.1033,
        -8.1241,
    ]
).view(-1, 1)
MELS_STD = torch.FloatTensor(
    [
        0.9299,
        1.5155,
        2.0167,
        2.1876,
        2.4266,
        2.4866,
        2.3538,
        2.2355,
        2.3034,
        2.4023,
        2.4576,
        2.4637,
        2.4387,
        2.4046,
        2.3737,
        2.3408,
        2.3098,
        2.2820,
        2.2551,
        2.2273,
        2.1941,
        2.1575,
        2.1280,
        2.1039,
        2.0897,
        2.0780,
        2.0685,
        2.0580,
        2.0497,
        2.0408,
        2.0356,
        2.0323,
        2.0293,
        2.0256,
        2.0248,
        2.0213,
        2.0233,
        2.0277,
        2.0288,
        2.0261,
        2.0189,
        2.0090,
        1.9984,
        1.9861,
        1.9764,
        1.9646,
        1.9553,
        1.9582,
        1.9724,
        1.9947,
        2.0112,
        2.0148,
        2.0057,
        1.9901,
        1.9714,
        1.9549,
        1.9396,
        1.9234,
        1.9077,
        1.8940,
        1.8857,
        1.8780,
        1.8678,
        1.8545,
        1.8406,
        1.8288,
        1.8193,
        1.8110,
        1.8053,
        1.8006,
        1.7950,
        1.7874,
        1.7783,
        1.7695,
        1.7583,
        1.7457,
        1.7357,
        1.7269,
        1.7246,
        1.7272,
    ]
).view(-1, 1)