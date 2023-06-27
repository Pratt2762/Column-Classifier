from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_2',
    'avg_pool_2',
    'skip_connect',
    'sep_conv_1',
    'sep_conv_3',
    'dil_conv_1',
    'dil_conv_3'
]

NASNet = Genotype(
    normal = [
        ('sep_conv_3', 1),
        ('sep_conv_1', 0),
        ('sep_conv_3', 0),
        ('sep_conv_1', 0),
        ('avg_pool_2', 1),
        ('skip_connect', 0),
        ('avg_pool_2', 0),
        ('avg_pool_2', 0),
        ('sep_conv_1', 1),
        ('skip_connect', 1),
    ],
    normal_concat = [2, 3, 4, 5, 6],
    reduce = [
        ('sep_conv_3', 1),
        ('sep_conv_5', 0),
        ('max_pool_2', 1),
        ('sep_conv_5', 0),
        ('avg_pool_2', 1),
        ('sep_conv_3', 0),
        ('skip_connect', 3),
        ('avg_pool_2', 2),
        ('sep_conv_1', 2),
        ('max_pool_2', 1),
    ],
    reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
    normal = [
        ('avg_pool_2', 0),
        ('max_pool_2', 1),
        ('sep_conv_1', 0),
        ('sep_conv_3', 2),
        ('sep_conv_1', 0),
        ('avg_pool_2', 3),
        ('sep_conv_1', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_2', 1),
    ],
    normal_concat = [4, 5, 6],
    reduce = [
        ('avg_pool_2', 0),
        ('sep_conv_1', 1),
        ('max_pool_2', 0),
        ('sep_conv_5', 2),
        ('sep_conv_5', 0),
        ('avg_pool_2', 1),
        ('max_pool_2', 0),
        ('max_pool_2', 1),
        ('sep_conv_1', 5),
    ],
    reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_1', 1), ('sep_conv_1', 0), ('skip_connect', 0), ('sep_conv_1', 1), ('skip_connect', 0), ('sep_conv_1', 1), ('sep_conv_1', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_2', 0), ('max_pool_2', 1), ('skip_connect', 2), ('max_pool_2', 0), ('max_pool_2', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_2', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_1', 0), ('sep_conv_1', 1), ('sep_conv_1', 0), ('sep_conv_1', 1), ('sep_conv_1', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_1', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_2', 0), ('max_pool_2', 1), ('skip_connect', 2), ('max_pool_2', 1), ('max_pool_2', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_2', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2
