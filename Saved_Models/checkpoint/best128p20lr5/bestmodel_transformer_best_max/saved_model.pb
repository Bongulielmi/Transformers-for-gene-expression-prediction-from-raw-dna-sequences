��0
�!�!
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
�
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.1-0-g85c8b2a817f8��)

conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �* 
shared_nameconv1d_4/kernel
x
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*#
_output_shapes
: �*
dtype0
s
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv1d_4/bias
l
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes	
:�*
dtype0

conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� * 
shared_nameconv1d_5/kernel
x
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*#
_output_shapes
:	� *
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
: *
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
: *
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	�@*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:@*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:@@*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:@*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:@*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:*
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
�
5token_and_position_embedding_2/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_2/embedding_4/embeddings
�
Itoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_4/embeddings*
_output_shapes

: *
dtype0
�
5token_and_position_embedding_2/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�R *F
shared_name75token_and_position_embedding_2/embedding_5/embeddings
�
Itoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_5/embeddings*
_output_shapes
:	�R *
dtype0
�
7transformer_block_5/multi_head_attention_5/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_5/multi_head_attention_5/query/kernel
�
Ktransformer_block_5/multi_head_attention_5/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_5/multi_head_attention_5/query/kernel*"
_output_shapes
:  *
dtype0
�
5transformer_block_5/multi_head_attention_5/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_5/multi_head_attention_5/query/bias
�
Itransformer_block_5/multi_head_attention_5/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_5/multi_head_attention_5/query/bias*
_output_shapes

: *
dtype0
�
5transformer_block_5/multi_head_attention_5/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_5/multi_head_attention_5/key/kernel
�
Itransformer_block_5/multi_head_attention_5/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_5/multi_head_attention_5/key/kernel*"
_output_shapes
:  *
dtype0
�
3transformer_block_5/multi_head_attention_5/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_5/multi_head_attention_5/key/bias
�
Gtransformer_block_5/multi_head_attention_5/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_5/multi_head_attention_5/key/bias*
_output_shapes

: *
dtype0
�
7transformer_block_5/multi_head_attention_5/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_5/multi_head_attention_5/value/kernel
�
Ktransformer_block_5/multi_head_attention_5/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_5/multi_head_attention_5/value/kernel*"
_output_shapes
:  *
dtype0
�
5transformer_block_5/multi_head_attention_5/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_5/multi_head_attention_5/value/bias
�
Itransformer_block_5/multi_head_attention_5/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_5/multi_head_attention_5/value/bias*
_output_shapes

: *
dtype0
�
Btransformer_block_5/multi_head_attention_5/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_5/multi_head_attention_5/attention_output/kernel
�
Vtransformer_block_5/multi_head_attention_5/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_5/multi_head_attention_5/attention_output/kernel*"
_output_shapes
:  *
dtype0
�
@transformer_block_5/multi_head_attention_5/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_5/multi_head_attention_5/attention_output/bias
�
Ttransformer_block_5/multi_head_attention_5/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_5/multi_head_attention_5/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

: @*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@ *
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
: *
dtype0
�
0transformer_block_5/layer_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_5/layer_normalization_10/gamma
�
Dtransformer_block_5/layer_normalization_10/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_5/layer_normalization_10/gamma*
_output_shapes
: *
dtype0
�
/transformer_block_5/layer_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_5/layer_normalization_10/beta
�
Ctransformer_block_5/layer_normalization_10/beta/Read/ReadVariableOpReadVariableOp/transformer_block_5/layer_normalization_10/beta*
_output_shapes
: *
dtype0
�
0transformer_block_5/layer_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_5/layer_normalization_11/gamma
�
Dtransformer_block_5/layer_normalization_11/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_5/layer_normalization_11/gamma*
_output_shapes
: *
dtype0
�
/transformer_block_5/layer_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_5/layer_normalization_11/beta
�
Ctransformer_block_5/layer_normalization_11/beta/Read/ReadVariableOpReadVariableOp/transformer_block_5/layer_normalization_11/beta*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
SGD/conv1d_4/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*-
shared_nameSGD/conv1d_4/kernel/momentum
�
0SGD/conv1d_4/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_4/kernel/momentum*#
_output_shapes
: �*
dtype0
�
SGD/conv1d_4/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameSGD/conv1d_4/bias/momentum
�
.SGD/conv1d_4/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_4/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGD/conv1d_5/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *-
shared_nameSGD/conv1d_5/kernel/momentum
�
0SGD/conv1d_5/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_5/kernel/momentum*#
_output_shapes
:	� *
dtype0
�
SGD/conv1d_5/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_5/bias/momentum
�
.SGD/conv1d_5/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_5/bias/momentum*
_output_shapes
: *
dtype0
�
(SGD/batch_normalization_4/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_4/gamma/momentum
�
<SGD/batch_normalization_4/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_4/gamma/momentum*
_output_shapes
: *
dtype0
�
'SGD/batch_normalization_4/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_4/beta/momentum
�
;SGD/batch_normalization_4/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_4/beta/momentum*
_output_shapes
: *
dtype0
�
(SGD/batch_normalization_5/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_5/gamma/momentum
�
<SGD/batch_normalization_5/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_5/gamma/momentum*
_output_shapes
: *
dtype0
�
'SGD/batch_normalization_5/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_5/beta/momentum
�
;SGD/batch_normalization_5/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_5/beta/momentum*
_output_shapes
: *
dtype0
�
SGD/dense_18/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*-
shared_nameSGD/dense_18/kernel/momentum
�
0SGD/dense_18/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_18/kernel/momentum*
_output_shapes
:	�@*
dtype0
�
SGD/dense_18/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_18/bias/momentum
�
.SGD/dense_18/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_18/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_19/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_19/kernel/momentum
�
0SGD/dense_19/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_19/kernel/momentum*
_output_shapes

:@@*
dtype0
�
SGD/dense_19/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_19/bias/momentum
�
.SGD/dense_19/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_19/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_20/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_20/kernel/momentum
�
0SGD/dense_20/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_20/kernel/momentum*
_output_shapes

:@*
dtype0
�
SGD/dense_20/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_20/bias/momentum
�
.SGD/dense_20/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_20/bias/momentum*
_output_shapes
:*
dtype0
�
BSGD/token_and_position_embedding_2/embedding_4/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum
�
VSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum*
_output_shapes

: *
dtype0
�
BSGD/token_and_position_embedding_2/embedding_5/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�R *S
shared_nameDBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum
�
VSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum*
_output_shapes
:	�R *
dtype0
�
DSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum
�
XSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum*"
_output_shapes
:  *
dtype0
�
BSGD/transformer_block_5/multi_head_attention_5/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum
�
VSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum*
_output_shapes

: *
dtype0
�
BSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum
�
VSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum*"
_output_shapes
:  *
dtype0
�
@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentum
�
TSGD/transformer_block_5/multi_head_attention_5/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentum*
_output_shapes

: *
dtype0
�
DSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum
�
XSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum*"
_output_shapes
:  *
dtype0
�
BSGD/transformer_block_5/multi_head_attention_5/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum
�
VSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum*
_output_shapes

: *
dtype0
�
OSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum
�
cSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
�
MSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum
�
aSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum*
_output_shapes
: *
dtype0
�
SGD/dense_16/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_16/kernel/momentum
�
0SGD/dense_16/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_16/kernel/momentum*
_output_shapes

: @*
dtype0
�
SGD/dense_16/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_16/bias/momentum
�
.SGD/dense_16/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_16/bias/momentum*
_output_shapes
:@*
dtype0
�
SGD/dense_17/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_17/kernel/momentum
�
0SGD/dense_17/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_17/kernel/momentum*
_output_shapes

:@ *
dtype0
�
SGD/dense_17/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_17/bias/momentum
�
.SGD/dense_17/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_17/bias/momentum*
_output_shapes
: *
dtype0
�
=SGD/transformer_block_5/layer_normalization_10/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_5/layer_normalization_10/gamma/momentum
�
QSGD/transformer_block_5/layer_normalization_10/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_5/layer_normalization_10/gamma/momentum*
_output_shapes
: *
dtype0
�
<SGD/transformer_block_5/layer_normalization_10/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_5/layer_normalization_10/beta/momentum
�
PSGD/transformer_block_5/layer_normalization_10/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_5/layer_normalization_10/beta/momentum*
_output_shapes
: *
dtype0
�
=SGD/transformer_block_5/layer_normalization_11/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_5/layer_normalization_11/gamma/momentum
�
QSGD/transformer_block_5/layer_normalization_11/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_5/layer_normalization_11/gamma/momentum*
_output_shapes
: *
dtype0
�
<SGD/transformer_block_5/layer_normalization_11/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_5/layer_normalization_11/beta/momentum
�
PSGD/transformer_block_5/layer_normalization_11/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_5/layer_normalization_11/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B޲ Bֲ
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
n
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
R
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api
�
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
�
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
�
Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
R
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
 
R
\	variables
]trainable_variables
^regularization_losses
_	keras_api
h

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
h

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
�
	zdecay
{learning_rate
|momentum
}iter momentum�!momentum�*momentum�+momentum�9momentum�:momentum�Bmomentum�Cmomentum�`momentum�amomentum�jmomentum�kmomentum�tmomentum�umomentum�~momentum�momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum�
�
~0
1
 2
!3
*4
+5
96
:7
;8
<9
B10
C11
D12
E13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
`30
a31
j32
k33
t34
u35
�
~0
1
 2
!3
*4
+5
96
:7
B8
C9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
`26
a27
j28
k29
t30
u31
 
�
	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
trainable_variables
regularization_losses
 
f
~
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
f

embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api

~0
1

~0
1
 
�
	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEconv1d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
�
"	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
#trainable_variables
$regularization_losses
 
 
 
�
&	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
'trainable_variables
(regularization_losses
[Y
VARIABLE_VALUEconv1d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
�
,	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
-trainable_variables
.regularization_losses
 
 
 
�
0	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
1trainable_variables
2regularization_losses
 
 
 
�
4	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
5trainable_variables
6regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

90
:1
;2
<3

90
:1
 
�
=	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
>trainable_variables
?regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2
E3

B0
C1
 
�
F	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Gtrainable_variables
Hregularization_losses
 
 
 
�
J	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Ktrainable_variables
Lregularization_losses
�
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
x
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
x
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
 
�
T	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Utrainable_variables
Vregularization_losses
 
 
 
�
X	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Ytrainable_variables
Zregularization_losses
 
 
 
�
\	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
]trainable_variables
^regularization_losses
[Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

`0
a1
 
�
b	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
ctrainable_variables
dregularization_losses
 
 
 
�
f	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
gtrainable_variables
hregularization_losses
[Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_19/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
�
l	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
mtrainable_variables
nregularization_losses
 
 
 
�
p	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
qtrainable_variables
rregularization_losses
[Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_20/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

t0
u1
 
�
v	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
wtrainable_variables
xregularization_losses
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_4/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_5/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_5/multi_head_attention_5/query/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/query/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/key/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE3transformer_block_5/multi_head_attention_5/key/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE7transformer_block_5/multi_head_attention_5/value/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE5transformer_block_5/multi_head_attention_5/value/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEBtransformer_block_5/multi_head_attention_5/attention_output/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE@transformer_block_5/multi_head_attention_5/attention_output/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_16/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_16/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_17/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_17/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_5/layer_normalization_10/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_5/layer_normalization_10/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE0transformer_block_5/layer_normalization_11/gamma'variables/28/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_5/layer_normalization_11/beta'variables/29/.ATTRIBUTES/VARIABLE_VALUE

�0

;0
<1
D2
E3
 
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18

~0

~0
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses

0

0
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
@
�0
�1
�2
�3
�4
�5
�6
�7
@
�0
�1
�2
�3
�4
�5
�6
�7
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
n
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
 
�0
�1
�2
�3
 
�0
�1
�2
�3
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
 
*
N0
O1
P2
Q3
R4
S5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
 
0
�0
�1
�2
�3
�4
�5

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses

�0
�1

�0
�1
 
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
��
VARIABLE_VALUESGD/conv1d_4/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv1d_4/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv1d_5/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/conv1d_5/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/batch_normalization_4/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'SGD/batch_normalization_4/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE(SGD/batch_normalization_5/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'SGD/batch_normalization_5/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_18/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_18/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_19/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_19/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_20/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUESGD/dense_20/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentumJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentumJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentumJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentumJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentumJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentumJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentumJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentumJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_16/kernel/momentumJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_16/bias/momentumJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUESGD/dense_17/kernel/momentumJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUESGD/dense_17/bias/momentumJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=SGD/transformer_block_5/layer_normalization_10/gamma/momentumJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<SGD/transformer_block_5/layer_normalization_10/beta/momentumJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE=SGD/transformer_block_5/layer_normalization_11/gamma/momentumJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<SGD/transformer_block_5/layer_normalization_11/beta/momentumJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_5Placeholder*(
_output_shapes
:����������R*
dtype0*
shape:����������R
z
serving_default_input_6Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5serving_default_input_65token_and_position_embedding_2/embedding_5/embeddings5token_and_position_embedding_2/embedding_4/embeddingsconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/beta%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/beta7transformer_block_5/multi_head_attention_5/query/kernel5transformer_block_5/multi_head_attention_5/query/bias5transformer_block_5/multi_head_attention_5/key/kernel3transformer_block_5/multi_head_attention_5/key/bias7transformer_block_5/multi_head_attention_5/value/kernel5transformer_block_5/multi_head_attention_5/value/biasBtransformer_block_5/multi_head_attention_5/attention_output/kernel@transformer_block_5/multi_head_attention_5/attention_output/bias0transformer_block_5/layer_normalization_10/gamma/transformer_block_5/layer_normalization_10/betadense_16/kerneldense_16/biasdense_17/kerneldense_17/bias0transformer_block_5/layer_normalization_11/gamma/transformer_block_5/layer_normalization_11/betadense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_186463
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpKtransformer_block_5/multi_head_attention_5/query/kernel/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/query/bias/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/key/kernel/Read/ReadVariableOpGtransformer_block_5/multi_head_attention_5/key/bias/Read/ReadVariableOpKtransformer_block_5/multi_head_attention_5/value/kernel/Read/ReadVariableOpItransformer_block_5/multi_head_attention_5/value/bias/Read/ReadVariableOpVtransformer_block_5/multi_head_attention_5/attention_output/kernel/Read/ReadVariableOpTtransformer_block_5/multi_head_attention_5/attention_output/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpDtransformer_block_5/layer_normalization_10/gamma/Read/ReadVariableOpCtransformer_block_5/layer_normalization_10/beta/Read/ReadVariableOpDtransformer_block_5/layer_normalization_11/gamma/Read/ReadVariableOpCtransformer_block_5/layer_normalization_11/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/conv1d_4/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_4/bias/momentum/Read/ReadVariableOp0SGD/conv1d_5/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_5/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_4/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_4/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_5/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_5/beta/momentum/Read/ReadVariableOp0SGD/dense_18/kernel/momentum/Read/ReadVariableOp.SGD/dense_18/bias/momentum/Read/ReadVariableOp0SGD/dense_19/kernel/momentum/Read/ReadVariableOp.SGD/dense_19/bias/momentum/Read/ReadVariableOp0SGD/dense_20/kernel/momentum/Read/ReadVariableOp.SGD/dense_20/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_5/multi_head_attention_5/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_16/kernel/momentum/Read/ReadVariableOp.SGD/dense_16/bias/momentum/Read/ReadVariableOp0SGD/dense_17/kernel/momentum/Read/ReadVariableOp.SGD/dense_17/bias/momentum/Read/ReadVariableOpQSGD/transformer_block_5/layer_normalization_10/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_5/layer_normalization_10/beta/momentum/Read/ReadVariableOpQSGD/transformer_block_5/layer_normalization_11/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_5/layer_normalization_11/beta/momentum/Read/ReadVariableOpConst*W
TinP
N2L	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_188546
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variancebatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variancedense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_2/embedding_4/embeddings5token_and_position_embedding_2/embedding_5/embeddings7transformer_block_5/multi_head_attention_5/query/kernel5transformer_block_5/multi_head_attention_5/query/bias5transformer_block_5/multi_head_attention_5/key/kernel3transformer_block_5/multi_head_attention_5/key/bias7transformer_block_5/multi_head_attention_5/value/kernel5transformer_block_5/multi_head_attention_5/value/biasBtransformer_block_5/multi_head_attention_5/attention_output/kernel@transformer_block_5/multi_head_attention_5/attention_output/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias0transformer_block_5/layer_normalization_10/gamma/transformer_block_5/layer_normalization_10/beta0transformer_block_5/layer_normalization_11/gamma/transformer_block_5/layer_normalization_11/betatotalcountSGD/conv1d_4/kernel/momentumSGD/conv1d_4/bias/momentumSGD/conv1d_5/kernel/momentumSGD/conv1d_5/bias/momentum(SGD/batch_normalization_4/gamma/momentum'SGD/batch_normalization_4/beta/momentum(SGD/batch_normalization_5/gamma/momentum'SGD/batch_normalization_5/beta/momentumSGD/dense_18/kernel/momentumSGD/dense_18/bias/momentumSGD/dense_19/kernel/momentumSGD/dense_19/bias/momentumSGD/dense_20/kernel/momentumSGD/dense_20/bias/momentumBSGD/token_and_position_embedding_2/embedding_4/embeddings/momentumBSGD/token_and_position_embedding_2/embedding_5/embeddings/momentumDSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentumBSGD/transformer_block_5/multi_head_attention_5/query/bias/momentumBSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentumDSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentumBSGD/transformer_block_5/multi_head_attention_5/value/bias/momentumOSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentumMSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentumSGD/dense_16/kernel/momentumSGD/dense_16/bias/momentumSGD/dense_17/kernel/momentumSGD/dense_17/bias/momentum=SGD/transformer_block_5/layer_normalization_10/gamma/momentum<SGD/transformer_block_5/layer_normalization_10/beta/momentum=SGD/transformer_block_5/layer_normalization_11/gamma/momentum<SGD/transformer_block_5/layer_normalization_11/beta/momentum*V
TinO
M2K*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_188778��&
��
�1
"__inference__traced_restore_188778
file_prefix$
 assignvariableop_conv1d_4_kernel$
 assignvariableop_1_conv1d_4_bias&
"assignvariableop_2_conv1d_5_kernel$
 assignvariableop_3_conv1d_5_bias2
.assignvariableop_4_batch_normalization_4_gamma1
-assignvariableop_5_batch_normalization_4_beta8
4assignvariableop_6_batch_normalization_4_moving_mean<
8assignvariableop_7_batch_normalization_4_moving_variance2
.assignvariableop_8_batch_normalization_5_gamma1
-assignvariableop_9_batch_normalization_5_beta9
5assignvariableop_10_batch_normalization_5_moving_mean=
9assignvariableop_11_batch_normalization_5_moving_variance'
#assignvariableop_12_dense_18_kernel%
!assignvariableop_13_dense_18_bias'
#assignvariableop_14_dense_19_kernel%
!assignvariableop_15_dense_19_bias'
#assignvariableop_16_dense_20_kernel%
!assignvariableop_17_dense_20_bias
assignvariableop_18_decay%
!assignvariableop_19_learning_rate 
assignvariableop_20_momentum 
assignvariableop_21_sgd_iterM
Iassignvariableop_22_token_and_position_embedding_2_embedding_4_embeddingsM
Iassignvariableop_23_token_and_position_embedding_2_embedding_5_embeddingsO
Kassignvariableop_24_transformer_block_5_multi_head_attention_5_query_kernelM
Iassignvariableop_25_transformer_block_5_multi_head_attention_5_query_biasM
Iassignvariableop_26_transformer_block_5_multi_head_attention_5_key_kernelK
Gassignvariableop_27_transformer_block_5_multi_head_attention_5_key_biasO
Kassignvariableop_28_transformer_block_5_multi_head_attention_5_value_kernelM
Iassignvariableop_29_transformer_block_5_multi_head_attention_5_value_biasZ
Vassignvariableop_30_transformer_block_5_multi_head_attention_5_attention_output_kernelX
Tassignvariableop_31_transformer_block_5_multi_head_attention_5_attention_output_bias'
#assignvariableop_32_dense_16_kernel%
!assignvariableop_33_dense_16_bias'
#assignvariableop_34_dense_17_kernel%
!assignvariableop_35_dense_17_biasH
Dassignvariableop_36_transformer_block_5_layer_normalization_10_gammaG
Cassignvariableop_37_transformer_block_5_layer_normalization_10_betaH
Dassignvariableop_38_transformer_block_5_layer_normalization_11_gammaG
Cassignvariableop_39_transformer_block_5_layer_normalization_11_beta
assignvariableop_40_total
assignvariableop_41_count4
0assignvariableop_42_sgd_conv1d_4_kernel_momentum2
.assignvariableop_43_sgd_conv1d_4_bias_momentum4
0assignvariableop_44_sgd_conv1d_5_kernel_momentum2
.assignvariableop_45_sgd_conv1d_5_bias_momentum@
<assignvariableop_46_sgd_batch_normalization_4_gamma_momentum?
;assignvariableop_47_sgd_batch_normalization_4_beta_momentum@
<assignvariableop_48_sgd_batch_normalization_5_gamma_momentum?
;assignvariableop_49_sgd_batch_normalization_5_beta_momentum4
0assignvariableop_50_sgd_dense_18_kernel_momentum2
.assignvariableop_51_sgd_dense_18_bias_momentum4
0assignvariableop_52_sgd_dense_19_kernel_momentum2
.assignvariableop_53_sgd_dense_19_bias_momentum4
0assignvariableop_54_sgd_dense_20_kernel_momentum2
.assignvariableop_55_sgd_dense_20_bias_momentumZ
Vassignvariableop_56_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentumZ
Vassignvariableop_57_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentum\
Xassignvariableop_58_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentumZ
Vassignvariableop_59_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentumZ
Vassignvariableop_60_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentumX
Tassignvariableop_61_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentum\
Xassignvariableop_62_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentumZ
Vassignvariableop_63_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentumg
cassignvariableop_64_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentume
aassignvariableop_65_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentum4
0assignvariableop_66_sgd_dense_16_kernel_momentum2
.assignvariableop_67_sgd_dense_16_bias_momentum4
0assignvariableop_68_sgd_dense_17_kernel_momentum2
.assignvariableop_69_sgd_dense_17_bias_momentumU
Qassignvariableop_70_sgd_transformer_block_5_layer_normalization_10_gamma_momentumT
Passignvariableop_71_sgd_transformer_block_5_layer_normalization_10_beta_momentumU
Qassignvariableop_72_sgd_transformer_block_5_layer_normalization_11_gamma_momentumT
Passignvariableop_73_sgd_transformer_block_5_layer_normalization_11_beta_momentum
identity_75��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_8�AssignVariableOp_9�%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�$
value�$B�$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�
value�B�KB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv1d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_4_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_4_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_4_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_4_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_5_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_5_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_5_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_5_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_18_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_18_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_19_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_19_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_20_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_20_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_sgd_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpIassignvariableop_22_token_and_position_embedding_2_embedding_4_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpIassignvariableop_23_token_and_position_embedding_2_embedding_5_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_5_multi_head_attention_5_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_5_multi_head_attention_5_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_5_multi_head_attention_5_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpGassignvariableop_27_transformer_block_5_multi_head_attention_5_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_5_multi_head_attention_5_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_5_multi_head_attention_5_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpVassignvariableop_30_transformer_block_5_multi_head_attention_5_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpTassignvariableop_31_transformer_block_5_multi_head_attention_5_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_16_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_16_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_17_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_17_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpDassignvariableop_36_transformer_block_5_layer_normalization_10_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpCassignvariableop_37_transformer_block_5_layer_normalization_10_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpDassignvariableop_38_transformer_block_5_layer_normalization_11_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOpCassignvariableop_39_transformer_block_5_layer_normalization_11_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp0assignvariableop_42_sgd_conv1d_4_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp.assignvariableop_43_sgd_conv1d_4_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_5_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_5_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp<assignvariableop_46_sgd_batch_normalization_4_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp;assignvariableop_47_sgd_batch_normalization_4_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_5_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_5_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_dense_18_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp.assignvariableop_51_sgd_dense_18_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_19_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_19_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_20_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_20_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOpVassignvariableop_56_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOpVassignvariableop_57_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOpXassignvariableop_58_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOpVassignvariableop_59_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOpTassignvariableop_61_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOpcassignvariableop_64_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOpaassignvariableop_65_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp0assignvariableop_66_sgd_dense_16_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp.assignvariableop_67_sgd_dense_16_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_dense_17_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_dense_17_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOpQassignvariableop_70_sgd_transformer_block_5_layer_normalization_10_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOpPassignvariableop_71_sgd_transformer_block_5_layer_normalization_10_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOpQassignvariableop_72_sgd_transformer_block_5_layer_normalization_11_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOpPassignvariableop_73_sgd_transformer_block_5_layer_normalization_11_beta_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_739
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_74�
Identity_75IdentityIdentity_74:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_75"#
identity_75Identity_75:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_185836

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_187743

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity��/layer_normalization_10/batchnorm/ReadVariableOp�3layer_normalization_10/batchnorm/mul/ReadVariableOp�/layer_normalization_11/batchnorm/ReadVariableOp�3layer_normalization_11/batchnorm/mul/ReadVariableOp�:multi_head_attention_5/attention_output/add/ReadVariableOp�Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_5/key/add/ReadVariableOp�7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/query/add/ReadVariableOp�9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/value/add/ReadVariableOp�9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�,sequential_5/dense_16/BiasAdd/ReadVariableOp�.sequential_5/dense_16/Tensordot/ReadVariableOp�,sequential_5/dense_17/BiasAdd/ReadVariableOp�.sequential_5/dense_17/Tensordot/ReadVariableOp�
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/Einsum�
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOp�
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/query/add�
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/Einsum�
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOp�
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2 
multi_head_attention_5/key/add�
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/Einsum�
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOp�
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/value/add�
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>2
multi_head_attention_5/Mul/y�
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 2
multi_head_attention_5/Mul�
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/Einsum�
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2(
&multi_head_attention_5/softmax/Softmax�
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,multi_head_attention_5/dropout/dropout/Const�
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������##2,
*multi_head_attention_5/dropout/dropout/Mul�
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_5/dropout/dropout/Shape�
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������##*
dtype0*

seedY2E
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform�
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_5/dropout/dropout/GreaterEqual/y�
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������##25
3multi_head_attention_5/dropout/dropout/GreaterEqual�
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������##2-
+multi_head_attention_5/dropout/dropout/Cast�
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������##2.
,multi_head_attention_5/dropout/dropout/Mul_1�
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsum�
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp�
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2-
+multi_head_attention_5/attention_output/addy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_14/dropout/Const�
dropout_14/dropout/MulMul/multi_head_attention_5/attention_output/add:z:0!dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:���������# 2
dropout_14/dropout/Mul�
dropout_14/dropout/ShapeShape/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape�
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:���������# *
dtype0*

seedY*
seed221
/dropout_14/dropout/random_uniform/RandomUniform�
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_14/dropout/GreaterEqual/y�
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������# 2!
dropout_14/dropout/GreaterEqual�
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������# 2
dropout_14/dropout/Cast�
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:���������# 2
dropout_14/dropout/Mul_1o
addAddV2inputsdropout_14/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������# 2
add�
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indices�
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_10/moments/mean�
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_10/moments/StopGradient�
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_10/moments/SquaredDifference�
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices�
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_10/moments/variance�
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_10/batchnorm/add/y�
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_10/batchnorm/add�
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_10/batchnorm/Rsqrt�
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOp�
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/mul�
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_1�
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_2�
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOp�
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/sub�
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/add_1�
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp�
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes�
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free�
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape�
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis�
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2�
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1�
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/Const�
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod�
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1�
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1�
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis�
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concat�
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stack�
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2+
)sequential_5/dense_16/Tensordot/transpose�
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_16/Tensordot/Reshape�
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2(
&sequential_5/dense_16/Tensordot/MatMul�
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2�
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis�
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1�
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2!
sequential_5/dense_16/Tensordot�
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp�
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/BiasAdd�
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/Relu�
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp�
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes�
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free�
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape�
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis�
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2�
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1�
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/Const�
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod�
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1�
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1�
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis�
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concat�
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stack�
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2+
)sequential_5/dense_17/Tensordot/transpose�
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_17/Tensordot/Reshape�
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&sequential_5/dense_17/Tensordot/MatMul�
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2�
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis�
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1�
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2!
sequential_5/dense_17/Tensordot�
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOp�
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2
sequential_5/dense_17/BiasAddy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_15/dropout/Const�
dropout_15/dropout/MulMul&sequential_5/dense_17/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:���������# 2
dropout_15/dropout/Mul�
dropout_15/dropout/ShapeShape&sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape�
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:���������# *
dtype0*

seedY*
seed221
/dropout_15/dropout/random_uniform/RandomUniform�
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_15/dropout/GreaterEqual/y�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������# 2!
dropout_15/dropout/GreaterEqual�
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������# 2
dropout_15/dropout/Cast�
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:���������# 2
dropout_15/dropout/Mul_1�
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������# 2
add_1�
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indices�
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_11/moments/mean�
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_11/moments/StopGradient�
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_11/moments/SquaredDifference�
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices�
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_11/moments/variance�
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_11/batchnorm/add/y�
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_11/batchnorm/add�
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_11/batchnorm/Rsqrt�
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOp�
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/mul�
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_1�
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_2�
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOp�
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/sub�
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/add_1�
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
��
�
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_185658

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity��/layer_normalization_10/batchnorm/ReadVariableOp�3layer_normalization_10/batchnorm/mul/ReadVariableOp�/layer_normalization_11/batchnorm/ReadVariableOp�3layer_normalization_11/batchnorm/mul/ReadVariableOp�:multi_head_attention_5/attention_output/add/ReadVariableOp�Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_5/key/add/ReadVariableOp�7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/query/add/ReadVariableOp�9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/value/add/ReadVariableOp�9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�,sequential_5/dense_16/BiasAdd/ReadVariableOp�.sequential_5/dense_16/Tensordot/ReadVariableOp�,sequential_5/dense_17/BiasAdd/ReadVariableOp�.sequential_5/dense_17/Tensordot/ReadVariableOp�
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/Einsum�
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOp�
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/query/add�
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/Einsum�
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOp�
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2 
multi_head_attention_5/key/add�
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/Einsum�
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOp�
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/value/add�
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>2
multi_head_attention_5/Mul/y�
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 2
multi_head_attention_5/Mul�
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/Einsum�
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2(
&multi_head_attention_5/softmax/Softmax�
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������##2)
'multi_head_attention_5/dropout/Identity�
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsum�
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp�
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2-
+multi_head_attention_5/attention_output/add�
dropout_14/IdentityIdentity/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:���������# 2
dropout_14/Identityo
addAddV2inputsdropout_14/Identity:output:0*
T0*+
_output_shapes
:���������# 2
add�
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indices�
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_10/moments/mean�
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_10/moments/StopGradient�
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_10/moments/SquaredDifference�
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices�
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_10/moments/variance�
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_10/batchnorm/add/y�
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_10/batchnorm/add�
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_10/batchnorm/Rsqrt�
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOp�
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/mul�
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_1�
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_2�
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOp�
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/sub�
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/add_1�
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp�
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes�
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free�
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape�
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis�
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2�
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1�
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/Const�
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod�
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1�
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1�
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis�
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concat�
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stack�
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2+
)sequential_5/dense_16/Tensordot/transpose�
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_16/Tensordot/Reshape�
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2(
&sequential_5/dense_16/Tensordot/MatMul�
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2�
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis�
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1�
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2!
sequential_5/dense_16/Tensordot�
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp�
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/BiasAdd�
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/Relu�
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp�
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes�
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free�
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape�
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis�
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2�
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1�
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/Const�
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod�
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1�
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1�
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis�
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concat�
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stack�
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2+
)sequential_5/dense_17/Tensordot/transpose�
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_17/Tensordot/Reshape�
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&sequential_5/dense_17/Tensordot/MatMul�
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2�
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis�
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1�
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2!
sequential_5/dense_17/Tensordot�
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOp�
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2
sequential_5/dense_17/BiasAdd�
dropout_15/IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:���������# 2
dropout_15/Identity�
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/Identity:output:0*
T0*+
_output_shapes
:���������# 2
add_1�
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indices�
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_11/moments/mean�
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_11/moments/StopGradient�
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_11/moments/SquaredDifference�
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices�
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_11/moments/variance�
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_11/batchnorm/add/y�
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_11/batchnorm/add�
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_11/batchnorm/Rsqrt�
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOp�
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/mul�
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_1�
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_2�
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOp�
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/sub�
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/add_1�
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_5_layer_call_fn_187583

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1853322
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185332

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187393

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
u
I__inference_concatenate_2_layer_call_and_return_conditional_losses_187962
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�0
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187373

inputs
assignmovingavg_187348
assignmovingavg_1_187354)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187348*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_187348*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187348*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187348*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_187348AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187348*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187354*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_187354*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187354*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187354*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_187354AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187354*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�0
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184723

inputs
assignmovingavg_184698
assignmovingavg_1_184704)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/184698*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184698*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/184698*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/184698*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184698AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/184698*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/184704*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184704*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184704*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184704*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184704AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/184704*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�	
�
D__inference_dense_20_layer_call_and_return_conditional_losses_185921

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
Z
.__inference_concatenate_2_layer_call_fn_187968
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_1857882
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':����������:���������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187557

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_188221

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1850632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
D__inference_conv1d_5_layer_call_and_return_conditional_losses_187246

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	� *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	� 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_188052

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�0
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187537

inputs
assignmovingavg_187512
assignmovingavg_1_187518)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187512*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_187512*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187512*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187512*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_187512AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187512*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187518*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_187518*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187518*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187518*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_187518AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187518*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
D__inference_conv1d_5_layer_call_and_return_conditional_losses_185168

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	� *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	� 2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_185773

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������# :S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
P
4__inference_average_pooling1d_7_layer_call_fn_184612

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_1846062
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
?__inference_token_and_position_embedding_2_layer_call_fn_187205
x
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_1851032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������R::22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:����������R

_user_specified_namex
� 
�
D__inference_dense_16_layer_call_and_return_conditional_losses_184942

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_187094
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1861302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�0
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_184863

inputs
assignmovingavg_184838
assignmovingavg_1_184844)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/184838*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_184838*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/184838*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/184838*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_184838AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/184838*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/184844*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_184844*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184844*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/184844*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_184844AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/184844*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�	
�
D__inference_dense_20_layer_call_and_return_conditional_losses_188072

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_185063

inputs
dense_16_185052
dense_16_185054
dense_17_185057
dense_17_185059
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_185052dense_16_185054*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1849422"
 dense_16/StatefulPartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_185057dense_17_185059*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1849882"
 dense_17/StatefulPartitionedCall�
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_185103
x'
#embedding_5_embedding_lookup_185090'
#embedding_4_embedding_lookup_185096
identity��embedding_4/embedding_lookup�embedding_5/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta�
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:���������2
range�
embedding_5/embedding_lookupResourceGather#embedding_5_embedding_lookup_185090range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_5/embedding_lookup/185090*'
_output_shapes
:��������� *
dtype02
embedding_5/embedding_lookup�
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_5/embedding_lookup/185090*'
_output_shapes
:��������� 2'
%embedding_5/embedding_lookup/Identity�
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� 2)
'embedding_5/embedding_lookup/Identity_1q
embedding_4/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:����������R2
embedding_4/Cast�
embedding_4/embedding_lookupResourceGather#embedding_4_embedding_lookup_185096embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_4/embedding_lookup/185096*,
_output_shapes
:����������R *
dtype02
embedding_4/embedding_lookup�
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_4/embedding_lookup/185096*,
_output_shapes
:����������R 2'
%embedding_4/embedding_lookup/Identity�
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������R 2)
'embedding_4/embedding_lookup/Identity_1�
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������R 2
add�
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:����������R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������R::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:����������R

_user_specified_namex
�
k
A__inference_add_2_layer_call_and_return_conditional_losses_185374

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:���������# :���������# :S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_188005

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
D__inference_dense_19_layer_call_and_return_conditional_losses_185865

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_185893

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
+__inference_dropout_17_layer_call_fn_188057

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_1858932
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
D__inference_dense_18_layer_call_and_return_conditional_losses_187979

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_184896

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
G
+__inference_dropout_16_layer_call_fn_188015

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_1858412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�&
C__inference_model_2_layer_call_and_return_conditional_losses_186773
inputs_0
inputs_1F
Btoken_and_position_embedding_2_embedding_5_embedding_lookup_186475F
Btoken_and_position_embedding_2_embedding_4_embedding_lookup_1864818
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource0
,batch_normalization_4_assignmovingavg_1865312
.batch_normalization_4_assignmovingavg_1_186537?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource0
,batch_normalization_5_assignmovingavg_1865632
.batch_normalization_5_assignmovingavg_1_186569?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resourceZ
Vtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_query_add_readvariableop_resourceX
Ttransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resourceZ
Vtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_value_add_readvariableop_resourcee
atransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity��9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_4/batchnorm/ReadVariableOp�2batch_normalization_4/batchnorm/mul/ReadVariableOp�9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_5/batchnorm/ReadVariableOp�2batch_normalization_5/batchnorm/mul/ReadVariableOp�conv1d_4/BiasAdd/ReadVariableOp�+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�conv1d_5/BiasAdd/ReadVariableOp�+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�;token_and_position_embedding_2/embedding_4/embedding_lookup�;token_and_position_embedding_2/embedding_5/embedding_lookup�Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp�Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp�Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp�Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp�Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp�Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp�Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp�Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp�Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp�Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp�@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp�Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp�
$token_and_position_embedding_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shape�
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������24
2token_and_position_embedding_2/strided_slice/stack�
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1�
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2�
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice�
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start�
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta�
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:���������2&
$token_and_position_embedding_2/range�
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_5_embedding_lookup_186475-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/186475*'
_output_shapes
:��������� *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup�
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/186475*'
_output_shapes
:��������� 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity�
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1�
/token_and_position_embedding_2/embedding_4/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:����������R21
/token_and_position_embedding_2/embedding_4/Cast�
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_4_embedding_lookup_1864813token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/186481*,
_output_shapes
:����������R *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup�
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/186481*,
_output_shapes
:����������R 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity�
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������R 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1�
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������R 2$
"token_and_position_embedding_2/add�
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_4/conv1d/ExpandDims/dim�
conv1d_4/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2
conv1d_4/conv1d/ExpandDims�
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim�
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: �2
conv1d_4/conv1d/ExpandDims_1�
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:����������R�*
paddingSAME*
strides
2
conv1d_4/conv1d�
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*-
_output_shapes
:����������R�*
squeeze_dims

���������2
conv1d_4/conv1d/Squeeze�
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp�
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:����������R�2
conv1d_4/BiasAddy
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*-
_output_shapes
:����������R�2
conv1d_4/Relu�
"average_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_6/ExpandDims/dim�
average_pooling1d_6/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0+average_pooling1d_6/ExpandDims/dim:output:0*
T0*1
_output_shapes
:����������R�2 
average_pooling1d_6/ExpandDims�
average_pooling1d_6/AvgPoolAvgPool'average_pooling1d_6/ExpandDims:output:0*
T0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
average_pooling1d_6/AvgPool�
average_pooling1d_6/SqueezeSqueeze$average_pooling1d_6/AvgPool:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims
2
average_pooling1d_6/Squeeze�
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_5/conv1d/ExpandDims/dim�
conv1d_5/conv1d/ExpandDims
ExpandDims$average_pooling1d_6/Squeeze:output:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������2
conv1d_5/conv1d/ExpandDims�
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	� *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim�
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	� 2
conv1d_5/conv1d/ExpandDims_1�
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d_5/conv1d�
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d_5/conv1d/Squeeze�
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_5/BiasAdd/ReadVariableOp�
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
conv1d_5/BiasAddx
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
conv1d_5/Relu�
"average_pooling1d_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_8/ExpandDims/dim�
average_pooling1d_8/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0+average_pooling1d_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2 
average_pooling1d_8/ExpandDims�
average_pooling1d_8/AvgPoolAvgPool'average_pooling1d_8/ExpandDims:output:0*
T0*/
_output_shapes
:���������# *
ksize	
�*
paddingVALID*
strides	
�2
average_pooling1d_8/AvgPool�
average_pooling1d_8/SqueezeSqueeze$average_pooling1d_8/AvgPool:output:0*
T0*+
_output_shapes
:���������# *
squeeze_dims
2
average_pooling1d_8/Squeeze�
"average_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_7/ExpandDims/dim�
average_pooling1d_7/ExpandDims
ExpandDimsconv1d_5/Relu:activations:0+average_pooling1d_7/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2 
average_pooling1d_7/ExpandDims�
average_pooling1d_7/AvgPoolAvgPool'average_pooling1d_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_7/AvgPool�
average_pooling1d_7/SqueezeSqueeze$average_pooling1d_7/AvgPool:output:0*
T0*+
_output_shapes
:���������# *
squeeze_dims
2
average_pooling1d_7/Squeeze�
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_4/moments/mean/reduction_indices�
"batch_normalization_4/moments/meanMean$average_pooling1d_7/Squeeze:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_4/moments/mean�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_4/moments/StopGradient�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifference$average_pooling1d_7/Squeeze:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 21
/batch_normalization_4/moments/SquaredDifference�
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_4/moments/variance/reduction_indices�
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_4/moments/variance�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze�
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1�
+batch_normalization_4/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/186531*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_4/AssignMovingAvg/decay�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_4_assignmovingavg_186531*
_output_shapes
: *
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/186531*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/sub�
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/186531*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/mul�
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_4_assignmovingavg_186531-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_4/AssignMovingAvg/186531*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_4/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/186537*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_4/AssignMovingAvg_1/decay�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_1_186537*
_output_shapes
: *
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/186537*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/sub�
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/186537*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/mul�
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_1_186537/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg_1/186537*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp�
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_4/batchnorm/add/y�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add�
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrt�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mul�
%batch_normalization_4/batchnorm/mul_1Mul$average_pooling1d_7/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_4/batchnorm/mul_1�
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp�
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/sub�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_4/batchnorm/add_1�
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_5/moments/mean/reduction_indices�
"batch_normalization_5/moments/meanMean$average_pooling1d_8/Squeeze:output:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_5/moments/mean�
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_5/moments/StopGradient�
/batch_normalization_5/moments/SquaredDifferenceSquaredDifference$average_pooling1d_8/Squeeze:output:03batch_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 21
/batch_normalization_5/moments/SquaredDifference�
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_5/moments/variance/reduction_indices�
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_5/moments/variance�
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze�
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1�
+batch_normalization_5/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/186563*
_output_shapes
: *
dtype0*
valueB
 *
�#<2-
+batch_normalization_5/AssignMovingAvg/decay�
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_5_assignmovingavg_186563*
_output_shapes
: *
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp�
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/186563*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/sub�
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/186563*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/mul�
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_5_assignmovingavg_186563-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_5/AssignMovingAvg/186563*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_5/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/186569*
_output_shapes
: *
dtype0*
valueB
 *
�#<2/
-batch_normalization_5/AssignMovingAvg_1/decay�
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_1_186569*
_output_shapes
: *
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/186569*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/sub�
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/186569*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/mul�
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_1_186569/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg_1/186569*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp�
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_5/batchnorm/add/y�
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/add�
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/Rsqrt�
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/mul�
%batch_normalization_5/batchnorm/mul_1Mul$average_pooling1d_8/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_5/batchnorm/mul_1�
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/mul_2�
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp�
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/sub�
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_5/batchnorm/add_1�
	add_2/addAddV2)batch_normalization_4/batchnorm/add_1:z:0)batch_normalization_5/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������# 2
	add_2/add�
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
>transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/query/einsum/Einsum�
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp�
4transformer_block_5/multi_head_attention_5/query/addAddV2Gtransformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 26
4transformer_block_5/multi_head_attention_5/query/add�
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
<transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2>
<transformer_block_5/multi_head_attention_5/key/einsum/Einsum�
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpJtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp�
2transformer_block_5/multi_head_attention_5/key/addAddV2Etransformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Itransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 24
2transformer_block_5/multi_head_attention_5/key/add�
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
>transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/value/einsum/Einsum�
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp�
4transformer_block_5/multi_head_attention_5/value/addAddV2Gtransformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 26
4transformer_block_5/multi_head_attention_5/value/add�
0transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>22
0transformer_block_5/multi_head_attention_5/Mul/y�
.transformer_block_5/multi_head_attention_5/MulMul8transformer_block_5/multi_head_attention_5/query/add:z:09transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 20
.transformer_block_5/multi_head_attention_5/Mul�
8transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum6transformer_block_5/multi_head_attention_5/key/add:z:02transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2:
8transformer_block_5/multi_head_attention_5/einsum/Einsum�
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxAtransformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2<
:transformer_block_5/multi_head_attention_5/softmax/Softmax�
@transformer_block_5/multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/Const�
>transformer_block_5/multi_head_attention_5/dropout/dropout/MulMulDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0Itransformer_block_5/multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������##2@
>transformer_block_5/multi_head_attention_5/dropout/dropout/Mul�
@transformer_block_5/multi_head_attention_5/dropout/dropout/ShapeShapeDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/Shape�
Wtransformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_5/multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������##*
dtype0*

seedY2Y
Wtransformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniform�
Itransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/y�
Gtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_5/multi_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������##2I
Gtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual�
?transformer_block_5/multi_head_attention_5/dropout/dropout/CastCastKtransformer_block_5/multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������##2A
?transformer_block_5/multi_head_attention_5/dropout/dropout/Cast�
@transformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1MulBtransformer_block_5/multi_head_attention_5/dropout/dropout/Mul:z:0Ctransformer_block_5/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������##2B
@transformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1�
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumDtransformer_block_5/multi_head_attention_5/dropout/dropout/Mul_1:z:08transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2<
:transformer_block_5/multi_head_attention_5/einsum_1/Einsum�
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumCtransformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0`transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe2K
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum�
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp�
?transformer_block_5/multi_head_attention_5/attention_output/addAddV2Rtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Vtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2A
?transformer_block_5/multi_head_attention_5/attention_output/add�
,transformer_block_5/dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2.
,transformer_block_5/dropout_14/dropout/Const�
*transformer_block_5/dropout_14/dropout/MulMulCtransformer_block_5/multi_head_attention_5/attention_output/add:z:05transformer_block_5/dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:���������# 2,
*transformer_block_5/dropout_14/dropout/Mul�
,transformer_block_5/dropout_14/dropout/ShapeShapeCtransformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_5/dropout_14/dropout/Shape�
Ctransformer_block_5/dropout_14/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_5/dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:���������# *
dtype0*

seedY*
seed22E
Ctransformer_block_5/dropout_14/dropout/random_uniform/RandomUniform�
5transformer_block_5/dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=27
5transformer_block_5/dropout_14/dropout/GreaterEqual/y�
3transformer_block_5/dropout_14/dropout/GreaterEqualGreaterEqualLtransformer_block_5/dropout_14/dropout/random_uniform/RandomUniform:output:0>transformer_block_5/dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������# 25
3transformer_block_5/dropout_14/dropout/GreaterEqual�
+transformer_block_5/dropout_14/dropout/CastCast7transformer_block_5/dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������# 2-
+transformer_block_5/dropout_14/dropout/Cast�
,transformer_block_5/dropout_14/dropout/Mul_1Mul.transformer_block_5/dropout_14/dropout/Mul:z:0/transformer_block_5/dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:���������# 2.
,transformer_block_5/dropout_14/dropout/Mul_1�
transformer_block_5/addAddV2add_2/add:z:00transformer_block_5/dropout_14/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������# 2
transformer_block_5/add�
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indices�
7transformer_block_5/layer_normalization_10/moments/meanMeantransformer_block_5/add:z:0Rtransformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(29
7transformer_block_5/layer_normalization_10/moments/mean�
?transformer_block_5/layer_normalization_10/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2A
?transformer_block_5/layer_normalization_10/moments/StopGradient�
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add:z:0Htransformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2F
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifference�
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices�
;transformer_block_5/layer_normalization_10/moments/varianceMeanHtransformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2=
;transformer_block_5/layer_normalization_10/moments/variance�
:transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52<
:transformer_block_5/layer_normalization_10/batchnorm/add/y�
8transformer_block_5/layer_normalization_10/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_10/moments/variance:output:0Ctransformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2:
8transformer_block_5/layer_normalization_10/batchnorm/add�
:transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2<
:transformer_block_5/layer_normalization_10/batchnorm/Rsqrt�
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp�
8transformer_block_5/layer_normalization_10/batchnorm/mulMul>transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_10/batchnorm/mul�
:transformer_block_5/layer_normalization_10/batchnorm/mul_1Multransformer_block_5/add:z:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_1�
:transformer_block_5/layer_normalization_10/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_10/moments/mean:output:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_2�
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp�
8transformer_block_5/layer_normalization_10/batchnorm/subSubKtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_10/batchnorm/sub�
:transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_10/batchnorm/add_1�
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp�
8transformer_block_5/sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_16/Tensordot/axes�
8transformer_block_5/sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_16/Tensordot/free�
9transformer_block_5/sequential_5/dense_16/Tensordot/ShapeShape>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/Shape�
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis�
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2�
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1�
9transformer_block_5/sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_16/Tensordot/Const�
8transformer_block_5/sequential_5/dense_16/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_16/Tensordot/Prod�
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1�
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1�
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis�
:transformer_block_5/sequential_5/dense_16/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_16/Tensordot/concat�
9transformer_block_5/sequential_5/dense_16/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_16/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/stack�
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose	Transpose>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2?
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose�
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Reshape�
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2<
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMul�
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2�
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis�
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1�
3transformer_block_5/sequential_5/dense_16/TensordotReshapeDtransformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@25
3transformer_block_5/sequential_5/dense_16/Tensordot�
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp�
1transformer_block_5/sequential_5/dense_16/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_16/Tensordot:output:0Htransformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@23
1transformer_block_5/sequential_5/dense_16/BiasAdd�
.transformer_block_5/sequential_5/dense_16/ReluRelu:transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@20
.transformer_block_5/sequential_5/dense_16/Relu�
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp�
8transformer_block_5/sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_17/Tensordot/axes�
8transformer_block_5/sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_17/Tensordot/free�
9transformer_block_5/sequential_5/dense_17/Tensordot/ShapeShape<transformer_block_5/sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/Shape�
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis�
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2�
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1�
9transformer_block_5/sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_17/Tensordot/Const�
8transformer_block_5/sequential_5/dense_17/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_17/Tensordot/Prod�
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1�
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1�
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis�
:transformer_block_5/sequential_5/dense_17/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_17/Tensordot/concat�
9transformer_block_5/sequential_5/dense_17/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_17/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/stack�
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose	Transpose<transformer_block_5/sequential_5/dense_16/Relu:activations:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2?
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose�
;transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_17/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Reshape�
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_17/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMul�
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2�
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis�
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1�
3transformer_block_5/sequential_5/dense_17/TensordotReshapeDtransformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 25
3transformer_block_5/sequential_5/dense_17/Tensordot�
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp�
1transformer_block_5/sequential_5/dense_17/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_17/Tensordot:output:0Htransformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 23
1transformer_block_5/sequential_5/dense_17/BiasAdd�
,transformer_block_5/dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2.
,transformer_block_5/dropout_15/dropout/Const�
*transformer_block_5/dropout_15/dropout/MulMul:transformer_block_5/sequential_5/dense_17/BiasAdd:output:05transformer_block_5/dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:���������# 2,
*transformer_block_5/dropout_15/dropout/Mul�
,transformer_block_5/dropout_15/dropout/ShapeShape:transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_5/dropout_15/dropout/Shape�
Ctransformer_block_5/dropout_15/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_5/dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:���������# *
dtype0*

seedY*
seed22E
Ctransformer_block_5/dropout_15/dropout/random_uniform/RandomUniform�
5transformer_block_5/dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=27
5transformer_block_5/dropout_15/dropout/GreaterEqual/y�
3transformer_block_5/dropout_15/dropout/GreaterEqualGreaterEqualLtransformer_block_5/dropout_15/dropout/random_uniform/RandomUniform:output:0>transformer_block_5/dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������# 25
3transformer_block_5/dropout_15/dropout/GreaterEqual�
+transformer_block_5/dropout_15/dropout/CastCast7transformer_block_5/dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������# 2-
+transformer_block_5/dropout_15/dropout/Cast�
,transformer_block_5/dropout_15/dropout/Mul_1Mul.transformer_block_5/dropout_15/dropout/Mul:z:0/transformer_block_5/dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:���������# 2.
,transformer_block_5/dropout_15/dropout/Mul_1�
transformer_block_5/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:00transformer_block_5/dropout_15/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������# 2
transformer_block_5/add_1�
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indices�
7transformer_block_5/layer_normalization_11/moments/meanMeantransformer_block_5/add_1:z:0Rtransformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(29
7transformer_block_5/layer_normalization_11/moments/mean�
?transformer_block_5/layer_normalization_11/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2A
?transformer_block_5/layer_normalization_11/moments/StopGradient�
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add_1:z:0Htransformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2F
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifference�
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices�
;transformer_block_5/layer_normalization_11/moments/varianceMeanHtransformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2=
;transformer_block_5/layer_normalization_11/moments/variance�
:transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52<
:transformer_block_5/layer_normalization_11/batchnorm/add/y�
8transformer_block_5/layer_normalization_11/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_11/moments/variance:output:0Ctransformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2:
8transformer_block_5/layer_normalization_11/batchnorm/add�
:transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2<
:transformer_block_5/layer_normalization_11/batchnorm/Rsqrt�
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp�
8transformer_block_5/layer_normalization_11/batchnorm/mulMul>transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_11/batchnorm/mul�
:transformer_block_5/layer_normalization_11/batchnorm/mul_1Multransformer_block_5/add_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_1�
:transformer_block_5/layer_normalization_11/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_11/moments/mean:output:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_2�
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp�
8transformer_block_5/layer_normalization_11/batchnorm/subSubKtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_11/batchnorm/sub�
:transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_11/batchnorm/add_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
flatten_2/Const�
flatten_2/ReshapeReshape>transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_2/Reshapex
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis�
concatenate_2/concatConcatV2flatten_2/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate_2/concat�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulconcatenate_2/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_18/Reluy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_16/dropout/Const�
dropout_16/dropout/MulMuldense_18/Relu:activations:0!dropout_16/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapedense_18/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape�
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedY*
seed221
/dropout_16/dropout/random_uniform/RandomUniform�
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2#
!dropout_16/dropout/GreaterEqual/y�
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_16/dropout/GreaterEqual�
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_16/dropout/Cast�
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_16/dropout/Mul_1�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_19/Reluy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_17/dropout/Const�
dropout_17/dropout/MulMuldense_19/Relu:activations:0!dropout_17/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape�
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedY*
seed221
/dropout_17/dropout/random_uniform/RandomUniform�
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2#
!dropout_17/dropout/GreaterEqual/y�
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2!
dropout_17/dropout/GreaterEqual�
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout_17/dropout/Cast�
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout_17/dropout/Mul_1�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd�
IdentityIdentitydense_20/BiasAdd:output:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp3^batch_normalization_5/batchnorm/mul/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupD^transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpD^transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpO^transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpY^transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpL^transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpA^transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpA^transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2�
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp2�
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp2�
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp2�
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp2�
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpNtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp2�
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2�
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpAtransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp2�
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpKtransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2�
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp2�
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2�
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp2�
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2�
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp2�
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp2�
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp2�
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:R N
(
_output_shapes
:����������R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
D__inference_dense_17_layer_call_and_return_conditional_losses_188291

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������#@
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_186205
input_5
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1861302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������R
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
��
�$
C__inference_model_2_layer_call_and_return_conditional_losses_187016
inputs_0
inputs_1F
Btoken_and_position_embedding_2_embedding_5_embedding_lookup_186785F
Btoken_and_position_embedding_2_embedding_4_embedding_lookup_1867918
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resourceZ
Vtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_query_add_readvariableop_resourceX
Ttransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resourceZ
Vtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_5_multi_head_attention_5_value_add_readvariableop_resourcee
atransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resourceO
Ktransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resourceM
Itransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resourceT
Ptransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource+
'dense_18_matmul_readvariableop_resource,
(dense_18_biasadd_readvariableop_resource+
'dense_19_matmul_readvariableop_resource,
(dense_19_biasadd_readvariableop_resource+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource
identity��.batch_normalization_4/batchnorm/ReadVariableOp�0batch_normalization_4/batchnorm/ReadVariableOp_1�0batch_normalization_4/batchnorm/ReadVariableOp_2�2batch_normalization_4/batchnorm/mul/ReadVariableOp�.batch_normalization_5/batchnorm/ReadVariableOp�0batch_normalization_5/batchnorm/ReadVariableOp_1�0batch_normalization_5/batchnorm/ReadVariableOp_2�2batch_normalization_5/batchnorm/mul/ReadVariableOp�conv1d_4/BiasAdd/ReadVariableOp�+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�conv1d_5/BiasAdd/ReadVariableOp�+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�;token_and_position_embedding_2/embedding_4/embedding_lookup�;token_and_position_embedding_2/embedding_5/embedding_lookup�Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp�Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp�Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp�Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp�Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp�Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp�Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp�Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp�Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp�Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp�@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp�Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp�
$token_and_position_embedding_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shape�
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������24
2token_and_position_embedding_2/strided_slice/stack�
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1�
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2�
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice�
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start�
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta�
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:���������2&
$token_and_position_embedding_2/range�
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_5_embedding_lookup_186785-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/186785*'
_output_shapes
:��������� *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup�
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/186785*'
_output_shapes
:��������� 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity�
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1�
/token_and_position_embedding_2/embedding_4/CastCastinputs_0*

DstT0*

SrcT0*(
_output_shapes
:����������R21
/token_and_position_embedding_2/embedding_4/Cast�
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherBtoken_and_position_embedding_2_embedding_4_embedding_lookup_1867913token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/186791*,
_output_shapes
:����������R *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup�
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/186791*,
_output_shapes
:����������R 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity�
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������R 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1�
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������R 2$
"token_and_position_embedding_2/add�
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_4/conv1d/ExpandDims/dim�
conv1d_4/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2
conv1d_4/conv1d/ExpandDims�
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dim�
conv1d_4/conv1d/ExpandDims_1
ExpandDims3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: �2
conv1d_4/conv1d/ExpandDims_1�
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:����������R�*
paddingSAME*
strides
2
conv1d_4/conv1d�
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*-
_output_shapes
:����������R�*
squeeze_dims

���������2
conv1d_4/conv1d/Squeeze�
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv1d_4/BiasAdd/ReadVariableOp�
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:����������R�2
conv1d_4/BiasAddy
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*-
_output_shapes
:����������R�2
conv1d_4/Relu�
"average_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_6/ExpandDims/dim�
average_pooling1d_6/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0+average_pooling1d_6/ExpandDims/dim:output:0*
T0*1
_output_shapes
:����������R�2 
average_pooling1d_6/ExpandDims�
average_pooling1d_6/AvgPoolAvgPool'average_pooling1d_6/ExpandDims:output:0*
T0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2
average_pooling1d_6/AvgPool�
average_pooling1d_6/SqueezeSqueeze$average_pooling1d_6/AvgPool:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims
2
average_pooling1d_6/Squeeze�
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
conv1d_5/conv1d/ExpandDims/dim�
conv1d_5/conv1d/ExpandDims
ExpandDims$average_pooling1d_6/Squeeze:output:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������2
conv1d_5/conv1d/ExpandDims�
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	� *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dim�
conv1d_5/conv1d/ExpandDims_1
ExpandDims3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	� 2
conv1d_5/conv1d/ExpandDims_1�
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
conv1d_5/conv1d�
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2
conv1d_5/conv1d/Squeeze�
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_5/BiasAdd/ReadVariableOp�
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
conv1d_5/BiasAddx
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
conv1d_5/Relu�
"average_pooling1d_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_8/ExpandDims/dim�
average_pooling1d_8/ExpandDims
ExpandDims&token_and_position_embedding_2/add:z:0+average_pooling1d_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2 
average_pooling1d_8/ExpandDims�
average_pooling1d_8/AvgPoolAvgPool'average_pooling1d_8/ExpandDims:output:0*
T0*/
_output_shapes
:���������# *
ksize	
�*
paddingVALID*
strides	
�2
average_pooling1d_8/AvgPool�
average_pooling1d_8/SqueezeSqueeze$average_pooling1d_8/AvgPool:output:0*
T0*+
_output_shapes
:���������# *
squeeze_dims
2
average_pooling1d_8/Squeeze�
"average_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_7/ExpandDims/dim�
average_pooling1d_7/ExpandDims
ExpandDimsconv1d_5/Relu:activations:0+average_pooling1d_7/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2 
average_pooling1d_7/ExpandDims�
average_pooling1d_7/AvgPoolAvgPool'average_pooling1d_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������# *
ksize

*
paddingVALID*
strides

2
average_pooling1d_7/AvgPool�
average_pooling1d_7/SqueezeSqueeze$average_pooling1d_7/AvgPool:output:0*
T0*+
_output_shapes
:���������# *
squeeze_dims
2
average_pooling1d_7/Squeeze�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp�
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_4/batchnorm/add/y�
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/add�
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/Rsqrt�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/mul�
%batch_normalization_4/batchnorm/mul_1Mul$average_pooling1d_7/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_4/batchnorm/mul_1�
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1�
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_4/batchnorm/mul_2�
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2�
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_4/batchnorm/sub�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_4/batchnorm/add_1�
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp�
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%batch_normalization_5/batchnorm/add/y�
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/add�
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/Rsqrt�
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp�
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/mul�
%batch_normalization_5/batchnorm/mul_1Mul$average_pooling1d_8/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_5/batchnorm/mul_1�
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1�
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_5/batchnorm/mul_2�
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2�
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_5/batchnorm/sub�
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2'
%batch_normalization_5/batchnorm/add_1�
	add_2/addAddV2)batch_normalization_4/batchnorm/add_1:z:0)batch_normalization_5/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������# 2
	add_2/add�
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
>transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/query/einsum/Einsum�
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp�
4transformer_block_5/multi_head_attention_5/query/addAddV2Gtransformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 26
4transformer_block_5/multi_head_attention_5/query/add�
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
<transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsumadd_2/add:z:0Stransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2>
<transformer_block_5/multi_head_attention_5/key/einsum/Einsum�
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpJtransformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp�
2transformer_block_5/multi_head_attention_5/key/addAddV2Etransformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Itransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 24
2transformer_block_5/multi_head_attention_5/key/add�
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
>transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsumadd_2/add:z:0Utransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2@
>transformer_block_5/multi_head_attention_5/value/einsum/Einsum�
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpLtransformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp�
4transformer_block_5/multi_head_attention_5/value/addAddV2Gtransformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Ktransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 26
4transformer_block_5/multi_head_attention_5/value/add�
0transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>22
0transformer_block_5/multi_head_attention_5/Mul/y�
.transformer_block_5/multi_head_attention_5/MulMul8transformer_block_5/multi_head_attention_5/query/add:z:09transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 20
.transformer_block_5/multi_head_attention_5/Mul�
8transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum6transformer_block_5/multi_head_attention_5/key/add:z:02transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2:
8transformer_block_5/multi_head_attention_5/einsum/Einsum�
:transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxAtransformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2<
:transformer_block_5/multi_head_attention_5/softmax/Softmax�
;transformer_block_5/multi_head_attention_5/dropout/IdentityIdentityDtransformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������##2=
;transformer_block_5/multi_head_attention_5/dropout/Identity�
:transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumDtransformer_block_5/multi_head_attention_5/dropout/Identity:output:08transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2<
:transformer_block_5/multi_head_attention_5/einsum_1/Einsum�
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumCtransformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0`transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe2K
Itransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum�
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp�
?transformer_block_5/multi_head_attention_5/attention_output/addAddV2Rtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0Vtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2A
?transformer_block_5/multi_head_attention_5/attention_output/add�
'transformer_block_5/dropout_14/IdentityIdentityCtransformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:���������# 2)
'transformer_block_5/dropout_14/Identity�
transformer_block_5/addAddV2add_2/add:z:00transformer_block_5/dropout_14/Identity:output:0*
T0*+
_output_shapes
:���������# 2
transformer_block_5/add�
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_10/moments/mean/reduction_indices�
7transformer_block_5/layer_normalization_10/moments/meanMeantransformer_block_5/add:z:0Rtransformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(29
7transformer_block_5/layer_normalization_10/moments/mean�
?transformer_block_5/layer_normalization_10/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2A
?transformer_block_5/layer_normalization_10/moments/StopGradient�
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add:z:0Htransformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2F
Dtransformer_block_5/layer_normalization_10/moments/SquaredDifference�
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices�
;transformer_block_5/layer_normalization_10/moments/varianceMeanHtransformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2=
;transformer_block_5/layer_normalization_10/moments/variance�
:transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52<
:transformer_block_5/layer_normalization_10/batchnorm/add/y�
8transformer_block_5/layer_normalization_10/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_10/moments/variance:output:0Ctransformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2:
8transformer_block_5/layer_normalization_10/batchnorm/add�
:transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2<
:transformer_block_5/layer_normalization_10/batchnorm/Rsqrt�
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp�
8transformer_block_5/layer_normalization_10/batchnorm/mulMul>transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_10/batchnorm/mul�
:transformer_block_5/layer_normalization_10/batchnorm/mul_1Multransformer_block_5/add:z:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_1�
:transformer_block_5/layer_normalization_10/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_10/moments/mean:output:0<transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_10/batchnorm/mul_2�
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp�
8transformer_block_5/layer_normalization_10/batchnorm/subSubKtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_10/batchnorm/sub�
:transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_10/batchnorm/add_1�
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp�
8transformer_block_5/sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_16/Tensordot/axes�
8transformer_block_5/sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_16/Tensordot/free�
9transformer_block_5/sequential_5/dense_16/Tensordot/ShapeShape>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/Shape�
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis�
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2�
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1�
9transformer_block_5/sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_16/Tensordot/Const�
8transformer_block_5/sequential_5/dense_16/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_16/Tensordot/Prod�
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_1�
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1�
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis�
:transformer_block_5/sequential_5/dense_16/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_16/Tensordot/concat�
9transformer_block_5/sequential_5/dense_16/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_16/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_16/Tensordot/stack�
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose	Transpose>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Ctransformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2?
=transformer_block_5/sequential_5/dense_16/Tensordot/transpose�
;transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Reshape�
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2<
:transformer_block_5/sequential_5/dense_16/Tensordot/MatMul�
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_5/sequential_5/dense_16/Tensordot/Const_2�
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis�
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_16/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_16/Tensordot/concat_1�
3transformer_block_5/sequential_5/dense_16/TensordotReshapeDtransformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@25
3transformer_block_5/sequential_5/dense_16/Tensordot�
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp�
1transformer_block_5/sequential_5/dense_16/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_16/Tensordot:output:0Htransformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@23
1transformer_block_5/sequential_5/dense_16/BiasAdd�
.transformer_block_5/sequential_5/dense_16/ReluRelu:transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@20
.transformer_block_5/sequential_5/dense_16/Relu�
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp�
8transformer_block_5/sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_5/sequential_5/dense_17/Tensordot/axes�
8transformer_block_5/sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_5/sequential_5/dense_17/Tensordot/free�
9transformer_block_5/sequential_5/dense_17/Tensordot/ShapeShape<transformer_block_5/sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/Shape�
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis�
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2�
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1GatherV2Btransformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Ltransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1�
9transformer_block_5/sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_5/sequential_5/dense_17/Tensordot/Const�
8transformer_block_5/sequential_5/dense_17/Tensordot/ProdProdEtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Btransformer_block_5/sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_5/sequential_5/dense_17/Tensordot/Prod�
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_1�
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1ProdGtransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1�
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis�
:transformer_block_5/sequential_5/dense_17/Tensordot/concatConcatV2Atransformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Atransformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Htransformer_block_5/sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_5/sequential_5/dense_17/Tensordot/concat�
9transformer_block_5/sequential_5/dense_17/Tensordot/stackPackAtransformer_block_5/sequential_5/dense_17/Tensordot/Prod:output:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_5/sequential_5/dense_17/Tensordot/stack�
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose	Transpose<transformer_block_5/sequential_5/dense_16/Relu:activations:0Ctransformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2?
=transformer_block_5/sequential_5/dense_17/Tensordot/transpose�
;transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeReshapeAtransformer_block_5/sequential_5/dense_17/Tensordot/transpose:y:0Btransformer_block_5/sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Reshape�
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMulMatMulDtransformer_block_5/sequential_5/dense_17/Tensordot/Reshape:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2<
:transformer_block_5/sequential_5/dense_17/Tensordot/MatMul�
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_5/sequential_5/dense_17/Tensordot/Const_2�
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis�
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1ConcatV2Etransformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Dtransformer_block_5/sequential_5/dense_17/Tensordot/Const_2:output:0Jtransformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_5/sequential_5/dense_17/Tensordot/concat_1�
3transformer_block_5/sequential_5/dense_17/TensordotReshapeDtransformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Etransformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 25
3transformer_block_5/sequential_5/dense_17/Tensordot�
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp�
1transformer_block_5/sequential_5/dense_17/BiasAddBiasAdd<transformer_block_5/sequential_5/dense_17/Tensordot:output:0Htransformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 23
1transformer_block_5/sequential_5/dense_17/BiasAdd�
'transformer_block_5/dropout_15/IdentityIdentity:transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:���������# 2)
'transformer_block_5/dropout_15/Identity�
transformer_block_5/add_1AddV2>transformer_block_5/layer_normalization_10/batchnorm/add_1:z:00transformer_block_5/dropout_15/Identity:output:0*
T0*+
_output_shapes
:���������# 2
transformer_block_5/add_1�
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_5/layer_normalization_11/moments/mean/reduction_indices�
7transformer_block_5/layer_normalization_11/moments/meanMeantransformer_block_5/add_1:z:0Rtransformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(29
7transformer_block_5/layer_normalization_11/moments/mean�
?transformer_block_5/layer_normalization_11/moments/StopGradientStopGradient@transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2A
?transformer_block_5/layer_normalization_11/moments/StopGradient�
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifferencetransformer_block_5/add_1:z:0Htransformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2F
Dtransformer_block_5/layer_normalization_11/moments/SquaredDifference�
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices�
;transformer_block_5/layer_normalization_11/moments/varianceMeanHtransformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0Vtransformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2=
;transformer_block_5/layer_normalization_11/moments/variance�
:transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52<
:transformer_block_5/layer_normalization_11/batchnorm/add/y�
8transformer_block_5/layer_normalization_11/batchnorm/addAddV2Dtransformer_block_5/layer_normalization_11/moments/variance:output:0Ctransformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2:
8transformer_block_5/layer_normalization_11/batchnorm/add�
:transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrt<transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2<
:transformer_block_5/layer_normalization_11/batchnorm/Rsqrt�
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp�
8transformer_block_5/layer_normalization_11/batchnorm/mulMul>transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Otransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_11/batchnorm/mul�
:transformer_block_5/layer_normalization_11/batchnorm/mul_1Multransformer_block_5/add_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_1�
:transformer_block_5/layer_normalization_11/batchnorm/mul_2Mul@transformer_block_5/layer_normalization_11/moments/mean:output:0<transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_11/batchnorm/mul_2�
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp�
8transformer_block_5/layer_normalization_11/batchnorm/subSubKtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0>transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2:
8transformer_block_5/layer_normalization_11/batchnorm/sub�
:transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2>transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0<transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2<
:transformer_block_5/layer_normalization_11/batchnorm/add_1s
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
flatten_2/Const�
flatten_2/ReshapeReshape>transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0flatten_2/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_2/Reshapex
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis�
concatenate_2/concatConcatV2flatten_2/Reshape:output:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatenate_2/concat�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMulconcatenate_2/concat:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_18/BiasAdds
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_18/Relu�
dropout_16/IdentityIdentitydense_18/Relu:activations:0*
T0*'
_output_shapes
:���������@2
dropout_16/Identity�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_19/MatMul/ReadVariableOp�
dense_19/MatMulMatMuldropout_16/Identity:output:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_19/MatMul�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_19/BiasAdd/ReadVariableOp�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_19/BiasAdds
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_19/Relu�
dropout_17/IdentityIdentitydense_19/Relu:activations:0*
T0*'
_output_shapes
:���������@2
dropout_17/Identity�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_20/MatMul/ReadVariableOp�
dense_20/MatMulMatMuldropout_17/Identity:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/MatMul�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_20/BiasAdd/ReadVariableOp�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_20/BiasAdd�
IdentityIdentitydense_20/BiasAdd:output:0/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp/^batch_normalization_5/batchnorm/ReadVariableOp1^batch_normalization_5/batchnorm/ReadVariableOp_11^batch_normalization_5/batchnorm/ReadVariableOp_23^batch_normalization_5/batchnorm/mul/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/conv1d/ExpandDims_1/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupD^transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpD^transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpH^transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpO^transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpY^transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpL^transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpD^transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpN^transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpA^transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpA^transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpC^transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp2`
.batch_normalization_5/batchnorm/ReadVariableOp.batch_normalization_5/batchnorm/ReadVariableOp2d
0batch_normalization_5/batchnorm/ReadVariableOp_10batch_normalization_5/batchnorm/ReadVariableOp_12d
0batch_normalization_5/batchnorm/ReadVariableOp_20batch_normalization_5/batchnorm/ReadVariableOp_22h
2batch_normalization_5/batchnorm/mul/ReadVariableOp2batch_normalization_5/batchnorm/mul/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp+conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp+conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2�
Ctransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp2�
Gtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp2�
Ctransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpCtransformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp2�
Gtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpGtransformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp2�
Ntransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpNtransformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp2�
Xtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2�
Atransformer_block_5/multi_head_attention_5/key/add/ReadVariableOpAtransformer_block_5/multi_head_attention_5/key/add/ReadVariableOp2�
Ktransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpKtransformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2�
Ctransformer_block_5/multi_head_attention_5/query/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/query/add/ReadVariableOp2�
Mtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2�
Ctransformer_block_5/multi_head_attention_5/value/add/ReadVariableOpCtransformer_block_5/multi_head_attention_5/value/add/ReadVariableOp2�
Mtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpMtransformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2�
@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp2�
Btransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp2�
@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp@transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp2�
Btransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpBtransformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:R N
(
_output_shapes
:����������R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_185241

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_187337

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1847562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�0
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187455

inputs
assignmovingavg_187430
assignmovingavg_1_187436)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187430*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_187430*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187430*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187430*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_187430AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187430*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187436*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_187436*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187436*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187436*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_187436AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187436*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
k
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_184606

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+���������������������������*
ksize

*
paddingVALID*
strides

2	
AvgPool�
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv1d_4_layer_call_and_return_conditional_losses_187221

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: �2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:����������R�*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:����������R�*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:����������R�2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:����������R�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:����������R�2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������R ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������R 
 
_user_specified_nameinputs
�
k
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_184621

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+���������������������������*
ksize	
�*
paddingVALID*
strides	
�2	
AvgPool�
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�J
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_188195

inputs.
*dense_16_tensordot_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource.
*dense_17_tensordot_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity��dense_16/BiasAdd/ReadVariableOp�!dense_16/Tensordot/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�!dense_17/Tensordot/ReadVariableOp�
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_16/Tensordot/ReadVariableOp|
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_16/Tensordot/axes�
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_16/Tensordot/freej
dense_16/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_16/Tensordot/Shape�
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/GatherV2/axis�
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2�
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_16/Tensordot/GatherV2_1/axis�
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2_1~
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const�
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod�
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const_1�
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod_1�
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_16/Tensordot/concat/axis�
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat�
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/stack�
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2
dense_16/Tensordot/transpose�
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_16/Tensordot/Reshape�
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_16/Tensordot/MatMul�
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_16/Tensordot/Const_2�
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/concat_1/axis�
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat_1�
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2
dense_16/Tensordot�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp�
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2
dense_16/BiasAddw
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
dense_16/Relu�
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_17/Tensordot/ReadVariableOp|
dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_17/Tensordot/axes�
dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_17/Tensordot/free
dense_17/Tensordot/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dense_17/Tensordot/Shape�
 dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/GatherV2/axis�
dense_17/Tensordot/GatherV2GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/free:output:0)dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2�
"dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_17/Tensordot/GatherV2_1/axis�
dense_17/Tensordot/GatherV2_1GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/axes:output:0+dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2_1~
dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const�
dense_17/Tensordot/ProdProd$dense_17/Tensordot/GatherV2:output:0!dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod�
dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_1�
dense_17/Tensordot/Prod_1Prod&dense_17/Tensordot/GatherV2_1:output:0#dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod_1�
dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_17/Tensordot/concat/axis�
dense_17/Tensordot/concatConcatV2 dense_17/Tensordot/free:output:0 dense_17/Tensordot/axes:output:0'dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat�
dense_17/Tensordot/stackPack dense_17/Tensordot/Prod:output:0"dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/stack�
dense_17/Tensordot/transpose	Transposedense_16/Relu:activations:0"dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2
dense_17/Tensordot/transpose�
dense_17/Tensordot/ReshapeReshape dense_17/Tensordot/transpose:y:0!dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_17/Tensordot/Reshape�
dense_17/Tensordot/MatMulMatMul#dense_17/Tensordot/Reshape:output:0)dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_17/Tensordot/MatMul�
dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_2�
 dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/concat_1/axis�
dense_17/Tensordot/concat_1ConcatV2$dense_17/Tensordot/GatherV2:output:0#dense_17/Tensordot/Const_2:output:0)dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat_1�
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2
dense_17/Tensordot�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOp�
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2
dense_17/BiasAdd�
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_187406

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1852212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�

�
4__inference_transformer_block_5_layer_call_fn_187907

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_1855312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_187172
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1863022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������R
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
d
+__inference_dropout_16_layer_call_fn_188010

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_1858362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_185841

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_185019
dense_16_input
dense_16_185008
dense_16_185010
dense_17_185013
dense_17_185015
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_185008dense_16_185010*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1849422"
 dense_16/StatefulPartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_185013dense_17_185015*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1849882"
 dense_17/StatefulPartitionedCall�
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:[ W
+
_output_shapes
:���������# 
(
_user_specified_namedense_16_input
�
P
4__inference_average_pooling1d_8_layer_call_fn_184627

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_1846212
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
�
D__inference_dense_18_layer_call_and_return_conditional_losses_185808

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_188208

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1850362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�

�
4__inference_transformer_block_5_layer_call_fn_187944

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_1856582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������# ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
D__inference_conv1d_4_layer_call_and_return_conditional_losses_185135

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv1d/ExpandDims/dim�
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2
conv1d/ExpandDims�
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim�
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: �2
conv1d/ExpandDims_1�
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:����������R�*
paddingSAME*
strides
2
conv1d�
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:����������R�*
squeeze_dims

���������2
conv1d/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:����������R�2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:����������R�2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:����������R�2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������R ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:����������R 
 
_user_specified_nameinputs
�
k
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_184591

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+���������������������������2

ExpandDims�
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+���������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'���������������������������*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
~
)__inference_dense_16_layer_call_fn_188261

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1849422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������# ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
Ϡ
�(
__inference__traced_save_188546
file_prefix.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop.
*savev2_dense_19_kernel_read_readvariableop,
(savev2_dense_19_bias_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopV
Rsavev2_transformer_block_5_multi_head_attention_5_query_kernel_read_readvariableopT
Psavev2_transformer_block_5_multi_head_attention_5_query_bias_read_readvariableopT
Psavev2_transformer_block_5_multi_head_attention_5_key_kernel_read_readvariableopR
Nsavev2_transformer_block_5_multi_head_attention_5_key_bias_read_readvariableopV
Rsavev2_transformer_block_5_multi_head_attention_5_value_kernel_read_readvariableopT
Psavev2_transformer_block_5_multi_head_attention_5_value_bias_read_readvariableopa
]savev2_transformer_block_5_multi_head_attention_5_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_5_multi_head_attention_5_attention_output_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableopO
Ksavev2_transformer_block_5_layer_normalization_10_gamma_read_readvariableopN
Jsavev2_transformer_block_5_layer_normalization_10_beta_read_readvariableopO
Ksavev2_transformer_block_5_layer_normalization_11_gamma_read_readvariableopN
Jsavev2_transformer_block_5_layer_normalization_11_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_conv1d_4_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_4_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_5_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_5_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_4_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_4_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_5_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_5_beta_momentum_read_readvariableop;
7savev2_sgd_dense_18_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_18_bias_momentum_read_readvariableop;
7savev2_sgd_dense_19_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_19_bias_momentum_read_readvariableop;
7savev2_sgd_dense_20_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_20_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_16_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_16_bias_momentum_read_readvariableop;
7savev2_sgd_dense_17_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_17_bias_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_5_layer_normalization_10_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_5_layer_normalization_10_beta_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_5_layer_normalization_11_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_5_layer_normalization_11_beta_momentum_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�$
value�$B�$KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/26/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/27/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/28/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/29/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*�
value�B�KB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop*savev2_dense_19_kernel_read_readvariableop(savev2_dense_19_bias_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopRsavev2_transformer_block_5_multi_head_attention_5_query_kernel_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_query_bias_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_key_kernel_read_readvariableopNsavev2_transformer_block_5_multi_head_attention_5_key_bias_read_readvariableopRsavev2_transformer_block_5_multi_head_attention_5_value_kernel_read_readvariableopPsavev2_transformer_block_5_multi_head_attention_5_value_bias_read_readvariableop]savev2_transformer_block_5_multi_head_attention_5_attention_output_kernel_read_readvariableop[savev2_transformer_block_5_multi_head_attention_5_attention_output_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableopKsavev2_transformer_block_5_layer_normalization_10_gamma_read_readvariableopJsavev2_transformer_block_5_layer_normalization_10_beta_read_readvariableopKsavev2_transformer_block_5_layer_normalization_11_gamma_read_readvariableopJsavev2_transformer_block_5_layer_normalization_11_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_4_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_4_bias_momentum_read_readvariableop7savev2_sgd_conv1d_5_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_5_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_4_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_4_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_5_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_5_beta_momentum_read_readvariableop7savev2_sgd_dense_18_kernel_momentum_read_readvariableop5savev2_sgd_dense_18_bias_momentum_read_readvariableop7savev2_sgd_dense_19_kernel_momentum_read_readvariableop5savev2_sgd_dense_19_bias_momentum_read_readvariableop7savev2_sgd_dense_20_kernel_momentum_read_readvariableop5savev2_sgd_dense_20_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_2_embedding_4_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_2_embedding_5_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_5_multi_head_attention_5_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_5_multi_head_attention_5_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_5_multi_head_attention_5_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_5_multi_head_attention_5_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_5_multi_head_attention_5_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_5_multi_head_attention_5_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_5_multi_head_attention_5_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_16_kernel_momentum_read_readvariableop5savev2_sgd_dense_16_bias_momentum_read_readvariableop7savev2_sgd_dense_17_kernel_momentum_read_readvariableop5savev2_sgd_dense_17_bias_momentum_read_readvariableopXsavev2_sgd_transformer_block_5_layer_normalization_10_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_5_layer_normalization_10_beta_momentum_read_readvariableopXsavev2_sgd_transformer_block_5_layer_normalization_11_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_5_layer_normalization_11_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : �:�:	� : : : : : : : : : :	�@:@:@@:@:@:: : : : : :	�R :  : :  : :  : :  : : @:@:@ : : : : : : : : �:�:	� : : : : : :	�@:@:@@:@:@:: :	�R :  : :  : :  : :  : : @:@:@ : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
: �:!

_output_shapes	
:�:)%
#
_output_shapes
:	� : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :%!

_output_shapes
:	�R :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :$ 

_output_shapes

: :($
"
_output_shapes
:  :  

_output_shapes
: :$! 

_output_shapes

: @: "

_output_shapes
:@:$# 

_output_shapes

:@ : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :)+%
#
_output_shapes
: �:!,

_output_shapes	
:�:)-%
#
_output_shapes
:	� : .

_output_shapes
: : /

_output_shapes
: : 0

_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: :%3!

_output_shapes
:	�@: 4

_output_shapes
:@:$5 

_output_shapes

:@@: 6

_output_shapes
:@:$7 

_output_shapes

:@: 8

_output_shapes
::$9 

_output_shapes

: :%:!

_output_shapes
:	�R :(;$
"
_output_shapes
:  :$< 

_output_shapes

: :(=$
"
_output_shapes
:  :$> 

_output_shapes

: :(?$
"
_output_shapes
:  :$@ 

_output_shapes

: :(A$
"
_output_shapes
:  : B

_output_shapes
: :$C 

_output_shapes

: @: D

_output_shapes
:@:$E 

_output_shapes

:@ : F

_output_shapes
: : G

_output_shapes
: : H

_output_shapes
: : I

_output_shapes
: : J

_output_shapes
: :K

_output_shapes
: 
�
G
+__inference_dropout_17_layer_call_fn_188062

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_1858982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_185036

inputs
dense_16_185025
dense_16_185027
dense_17_185030
dense_17_185032
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCallinputsdense_16_185025dense_16_185027*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1849422"
 dense_16/StatefulPartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_185030dense_17_185032*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1849882"
 dense_17/StatefulPartitionedCall�
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187311

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_186377
input_5
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_1863022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������R
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�
P
4__inference_average_pooling1d_6_layer_call_fn_184597

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'���������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_1845912
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'���������������������������2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�[
�
C__inference_model_2_layer_call_and_return_conditional_losses_186130

inputs
inputs_1)
%token_and_position_embedding_2_186040)
%token_and_position_embedding_2_186042
conv1d_4_186045
conv1d_4_186047
conv1d_5_186051
conv1d_5_186053 
batch_normalization_4_186058 
batch_normalization_4_186060 
batch_normalization_4_186062 
batch_normalization_4_186064 
batch_normalization_5_186067 
batch_normalization_5_186069 
batch_normalization_5_186071 
batch_normalization_5_186073
transformer_block_5_186077
transformer_block_5_186079
transformer_block_5_186081
transformer_block_5_186083
transformer_block_5_186085
transformer_block_5_186087
transformer_block_5_186089
transformer_block_5_186091
transformer_block_5_186093
transformer_block_5_186095
transformer_block_5_186097
transformer_block_5_186099
transformer_block_5_186101
transformer_block_5_186103
transformer_block_5_186105
transformer_block_5_186107
dense_18_186112
dense_18_186114
dense_19_186118
dense_19_186120
dense_20_186124
dense_20_186126
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�"dropout_17/StatefulPartitionedCall�6token_and_position_embedding_2/StatefulPartitionedCall�+transformer_block_5/StatefulPartitionedCall�
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_2_186040%token_and_position_embedding_2_186042*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_18510328
6token_and_position_embedding_2/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_186045conv1d_4_186047*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:����������R�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_1851352"
 conv1d_4/StatefulPartitionedCall�
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_1845912%
#average_pooling1d_6/PartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_186051conv1d_5_186053*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_1851682"
 conv1d_5/StatefulPartitionedCall�
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_1846212%
#average_pooling1d_8/PartitionedCall�
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_1846062%
#average_pooling1d_7/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_4_186058batch_normalization_4_186060batch_normalization_4_186062batch_normalization_4_186064*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1852212/
-batch_normalization_4/StatefulPartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_5_186067batch_normalization_5_186069batch_normalization_5_186071batch_normalization_5_186073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1853122/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1853742
add_2/PartitionedCall�
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_186077transformer_block_5_186079transformer_block_5_186081transformer_block_5_186083transformer_block_5_186085transformer_block_5_186087transformer_block_5_186089transformer_block_5_186091transformer_block_5_186093transformer_block_5_186095transformer_block_5_186097transformer_block_5_186099transformer_block_5_186101transformer_block_5_186103transformer_block_5_186105transformer_block_5_186107*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_1855312-
+transformer_block_5/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1857732
flatten_2/PartitionedCall�
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_1857882
concatenate_2/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_186112dense_18_186114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_1858082"
 dense_18/StatefulPartitionedCall�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_1858362$
"dropout_16/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_19_186118dense_19_186120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_1858652"
 dense_19/StatefulPartitionedCall�
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_1858932$
"dropout_17/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_20_186124dense_20_186126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1859212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:P L
(
_output_shapes
:����������R
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
R
&__inference_add_2_layer_call_fn_187595
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1853742
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:���������# :���������# :U Q
+
_output_shapes
:���������# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������# 
"
_user_specified_name
inputs/1
�0
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_185221

inputs
assignmovingavg_185196
assignmovingavg_1_185202)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/185196*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_185196*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/185196*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/185196*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_185196AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/185196*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/185202*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_185202*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/185202*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/185202*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_185202AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/185202*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_188047

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_flatten_2_layer_call_fn_187955

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1857732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������# :S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
s
I__inference_concatenate_2_layer_call_and_return_conditional_losses_185788

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':����������:���������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
~
)__inference_conv1d_4_layer_call_fn_187230

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:����������R�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_1851352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:����������R�2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :����������R ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������R 
 
_user_specified_nameinputs
��
�(
!__inference__wrapped_model_184582
input_5
input_6N
Jmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_184351N
Jmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_184357@
<model_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource4
0model_2_conv1d_4_biasadd_readvariableop_resource@
<model_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource4
0model_2_conv1d_5_biasadd_readvariableop_resourceC
?model_2_batch_normalization_4_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_4_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_4_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_4_batchnorm_readvariableop_2_resourceC
?model_2_batch_normalization_5_batchnorm_readvariableop_resourceG
Cmodel_2_batch_normalization_5_batchnorm_mul_readvariableop_resourceE
Amodel_2_batch_normalization_5_batchnorm_readvariableop_1_resourceE
Amodel_2_batch_normalization_5_batchnorm_readvariableop_2_resourceb
^model_2_transformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_5_multi_head_attention_5_query_add_readvariableop_resource`
\model_2_transformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resourceV
Rmodel_2_transformer_block_5_multi_head_attention_5_key_add_readvariableop_resourceb
^model_2_transformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resourceX
Tmodel_2_transformer_block_5_multi_head_attention_5_value_add_readvariableop_resourcem
imodel_2_transformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resourcec
_model_2_transformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource\
Xmodel_2_transformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resourceX
Tmodel_2_transformer_block_5_layer_normalization_10_batchnorm_readvariableop_resourceW
Smodel_2_transformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resourceU
Qmodel_2_transformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resourceW
Smodel_2_transformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resourceU
Qmodel_2_transformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource\
Xmodel_2_transformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resourceX
Tmodel_2_transformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource3
/model_2_dense_18_matmul_readvariableop_resource4
0model_2_dense_18_biasadd_readvariableop_resource3
/model_2_dense_19_matmul_readvariableop_resource4
0model_2_dense_19_biasadd_readvariableop_resource3
/model_2_dense_20_matmul_readvariableop_resource4
0model_2_dense_20_biasadd_readvariableop_resource
identity��6model_2/batch_normalization_4/batchnorm/ReadVariableOp�8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1�8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2�:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp�6model_2/batch_normalization_5/batchnorm/ReadVariableOp�8model_2/batch_normalization_5/batchnorm/ReadVariableOp_1�8model_2/batch_normalization_5/batchnorm/ReadVariableOp_2�:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp�'model_2/conv1d_4/BiasAdd/ReadVariableOp�3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�'model_2/conv1d_5/BiasAdd/ReadVariableOp�3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�'model_2/dense_18/BiasAdd/ReadVariableOp�&model_2/dense_18/MatMul/ReadVariableOp�'model_2/dense_19/BiasAdd/ReadVariableOp�&model_2/dense_19/MatMul/ReadVariableOp�'model_2/dense_20/BiasAdd/ReadVariableOp�&model_2/dense_20/MatMul/ReadVariableOp�Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup�Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup�Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp�Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp�Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp�Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp�Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp�`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp�Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp�Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp�Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp�Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp�Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp�Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp�
,model_2/token_and_position_embedding_2/ShapeShapeinput_5*
T0*
_output_shapes
:2.
,model_2/token_and_position_embedding_2/Shape�
:model_2/token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2<
:model_2/token_and_position_embedding_2/strided_slice/stack�
<model_2/token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_2/token_and_position_embedding_2/strided_slice/stack_1�
<model_2/token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/token_and_position_embedding_2/strided_slice/stack_2�
4model_2/token_and_position_embedding_2/strided_sliceStridedSlice5model_2/token_and_position_embedding_2/Shape:output:0Cmodel_2/token_and_position_embedding_2/strided_slice/stack:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_1:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/token_and_position_embedding_2/strided_slice�
2model_2/token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/token_and_position_embedding_2/range/start�
2model_2/token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/token_and_position_embedding_2/range/delta�
,model_2/token_and_position_embedding_2/rangeRange;model_2/token_and_position_embedding_2/range/start:output:0=model_2/token_and_position_embedding_2/strided_slice:output:0;model_2/token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:���������2.
,model_2/token_and_position_embedding_2/range�
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherJmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_1843515model_2/token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/184351*'
_output_shapes
:��������� *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup�
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/184351*'
_output_shapes
:��������� 2N
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity�
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� 2P
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1�
7model_2/token_and_position_embedding_2/embedding_4/CastCastinput_5*

DstT0*

SrcT0*(
_output_shapes
:����������R29
7model_2/token_and_position_embedding_2/embedding_4/Cast�
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherJmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_184357;model_2/token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/184357*,
_output_shapes
:����������R *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup�
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/184357*,
_output_shapes
:����������R 2N
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity�
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������R 2P
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1�
*model_2/token_and_position_embedding_2/addAddV2Wmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������R 2,
*model_2/token_and_position_embedding_2/add�
&model_2/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&model_2/conv1d_4/conv1d/ExpandDims/dim�
"model_2/conv1d_4/conv1d/ExpandDims
ExpandDims.model_2/token_and_position_embedding_2/add:z:0/model_2/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2$
"model_2/conv1d_4/conv1d/ExpandDims�
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_4_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype025
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp�
(model_2/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_4/conv1d/ExpandDims_1/dim�
$model_2/conv1d_4/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: �2&
$model_2/conv1d_4/conv1d/ExpandDims_1�
model_2/conv1d_4/conv1dConv2D+model_2/conv1d_4/conv1d/ExpandDims:output:0-model_2/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:����������R�*
paddingSAME*
strides
2
model_2/conv1d_4/conv1d�
model_2/conv1d_4/conv1d/SqueezeSqueeze model_2/conv1d_4/conv1d:output:0*
T0*-
_output_shapes
:����������R�*
squeeze_dims

���������2!
model_2/conv1d_4/conv1d/Squeeze�
'model_2/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_2/conv1d_4/BiasAdd/ReadVariableOp�
model_2/conv1d_4/BiasAddBiasAdd(model_2/conv1d_4/conv1d/Squeeze:output:0/model_2/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:����������R�2
model_2/conv1d_4/BiasAdd�
model_2/conv1d_4/ReluRelu!model_2/conv1d_4/BiasAdd:output:0*
T0*-
_output_shapes
:����������R�2
model_2/conv1d_4/Relu�
*model_2/average_pooling1d_6/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_6/ExpandDims/dim�
&model_2/average_pooling1d_6/ExpandDims
ExpandDims#model_2/conv1d_4/Relu:activations:03model_2/average_pooling1d_6/ExpandDims/dim:output:0*
T0*1
_output_shapes
:����������R�2(
&model_2/average_pooling1d_6/ExpandDims�
#model_2/average_pooling1d_6/AvgPoolAvgPool/model_2/average_pooling1d_6/ExpandDims:output:0*
T0*1
_output_shapes
:�����������*
ksize
*
paddingVALID*
strides
2%
#model_2/average_pooling1d_6/AvgPool�
#model_2/average_pooling1d_6/SqueezeSqueeze,model_2/average_pooling1d_6/AvgPool:output:0*
T0*-
_output_shapes
:�����������*
squeeze_dims
2%
#model_2/average_pooling1d_6/Squeeze�
&model_2/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&model_2/conv1d_5/conv1d/ExpandDims/dim�
"model_2/conv1d_5/conv1d/ExpandDims
ExpandDims,model_2/average_pooling1d_6/Squeeze:output:0/model_2/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������2$
"model_2/conv1d_5/conv1d/ExpandDims�
3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_2_conv1d_5_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:	� *
dtype025
3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp�
(model_2/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_2/conv1d_5/conv1d/ExpandDims_1/dim�
$model_2/conv1d_5/conv1d/ExpandDims_1
ExpandDims;model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:01model_2/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:	� 2&
$model_2/conv1d_5/conv1d/ExpandDims_1�
model_2/conv1d_5/conv1dConv2D+model_2/conv1d_5/conv1d/ExpandDims:output:0-model_2/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������� *
paddingSAME*
strides
2
model_2/conv1d_5/conv1d�
model_2/conv1d_5/conv1d/SqueezeSqueeze model_2/conv1d_5/conv1d:output:0*
T0*,
_output_shapes
:���������� *
squeeze_dims

���������2!
model_2/conv1d_5/conv1d/Squeeze�
'model_2/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp0model_2_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_2/conv1d_5/BiasAdd/ReadVariableOp�
model_2/conv1d_5/BiasAddBiasAdd(model_2/conv1d_5/conv1d/Squeeze:output:0/model_2/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������� 2
model_2/conv1d_5/BiasAdd�
model_2/conv1d_5/ReluRelu!model_2/conv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:���������� 2
model_2/conv1d_5/Relu�
*model_2/average_pooling1d_8/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_8/ExpandDims/dim�
&model_2/average_pooling1d_8/ExpandDims
ExpandDims.model_2/token_and_position_embedding_2/add:z:03model_2/average_pooling1d_8/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������R 2(
&model_2/average_pooling1d_8/ExpandDims�
#model_2/average_pooling1d_8/AvgPoolAvgPool/model_2/average_pooling1d_8/ExpandDims:output:0*
T0*/
_output_shapes
:���������# *
ksize	
�*
paddingVALID*
strides	
�2%
#model_2/average_pooling1d_8/AvgPool�
#model_2/average_pooling1d_8/SqueezeSqueeze,model_2/average_pooling1d_8/AvgPool:output:0*
T0*+
_output_shapes
:���������# *
squeeze_dims
2%
#model_2/average_pooling1d_8/Squeeze�
*model_2/average_pooling1d_7/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_7/ExpandDims/dim�
&model_2/average_pooling1d_7/ExpandDims
ExpandDims#model_2/conv1d_5/Relu:activations:03model_2/average_pooling1d_7/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������� 2(
&model_2/average_pooling1d_7/ExpandDims�
#model_2/average_pooling1d_7/AvgPoolAvgPool/model_2/average_pooling1d_7/ExpandDims:output:0*
T0*/
_output_shapes
:���������# *
ksize

*
paddingVALID*
strides

2%
#model_2/average_pooling1d_7/AvgPool�
#model_2/average_pooling1d_7/SqueezeSqueeze,model_2/average_pooling1d_7/AvgPool:output:0*
T0*+
_output_shapes
:���������# *
squeeze_dims
2%
#model_2/average_pooling1d_7/Squeeze�
6model_2/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_2/batch_normalization_4/batchnorm/ReadVariableOp�
-model_2/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2/
-model_2/batch_normalization_4/batchnorm/add/y�
+model_2/batch_normalization_4/batchnorm/addAddV2>model_2/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_4/batchnorm/add�
-model_2/batch_normalization_4/batchnorm/RsqrtRsqrt/model_2/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_4/batchnorm/Rsqrt�
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp�
+model_2/batch_normalization_4/batchnorm/mulMul1model_2/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_4/batchnorm/mul�
-model_2/batch_normalization_4/batchnorm/mul_1Mul,model_2/average_pooling1d_7/Squeeze:output:0/model_2/batch_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2/
-model_2/batch_normalization_4/batchnorm/mul_1�
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_1�
-model_2/batch_normalization_4/batchnorm/mul_2Mul@model_2/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_4/batchnorm/mul_2�
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_2�
+model_2/batch_normalization_4/batchnorm/subSub@model_2/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_4/batchnorm/sub�
-model_2/batch_normalization_4/batchnorm/add_1AddV21model_2/batch_normalization_4/batchnorm/mul_1:z:0/model_2/batch_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2/
-model_2/batch_normalization_4/batchnorm/add_1�
6model_2/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp?model_2_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_2/batch_normalization_5/batchnorm/ReadVariableOp�
-model_2/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2/
-model_2/batch_normalization_5/batchnorm/add/y�
+model_2/batch_normalization_5/batchnorm/addAddV2>model_2/batch_normalization_5/batchnorm/ReadVariableOp:value:06model_2/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_5/batchnorm/add�
-model_2/batch_normalization_5/batchnorm/RsqrtRsqrt/model_2/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_5/batchnorm/Rsqrt�
:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_2_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp�
+model_2/batch_normalization_5/batchnorm/mulMul1model_2/batch_normalization_5/batchnorm/Rsqrt:y:0Bmodel_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_5/batchnorm/mul�
-model_2/batch_normalization_5/batchnorm/mul_1Mul,model_2/average_pooling1d_8/Squeeze:output:0/model_2/batch_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2/
-model_2/batch_normalization_5/batchnorm/mul_1�
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_2_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_1�
-model_2/batch_normalization_5/batchnorm/mul_2Mul@model_2/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0/model_2/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_2/batch_normalization_5/batchnorm/mul_2�
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_2_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_2�
+model_2/batch_normalization_5/batchnorm/subSub@model_2/batch_normalization_5/batchnorm/ReadVariableOp_2:value:01model_2/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_2/batch_normalization_5/batchnorm/sub�
-model_2/batch_normalization_5/batchnorm/add_1AddV21model_2/batch_normalization_5/batchnorm/mul_1:z:0/model_2/batch_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2/
-model_2/batch_normalization_5/batchnorm/add_1�
model_2/add_2/addAddV21model_2/batch_normalization_4/batchnorm/add_1:z:01model_2/batch_normalization_5/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������# 2
model_2/add_2/add�
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_5_multi_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
Fmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2H
Fmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum�
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp�
<model_2/transformer_block_5/multi_head_attention_5/query/addAddV2Omodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum:output:0Smodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2>
<model_2/transformer_block_5/multi_head_attention_5/query/add�
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_2_transformer_block_5_multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
Dmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/EinsumEinsummodel_2/add_2/add:z:0[model_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2F
Dmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum�
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpReadVariableOpRmodel_2_transformer_block_5_multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp�
:model_2/transformer_block_5/multi_head_attention_5/key/addAddV2Mmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum:output:0Qmodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2<
:model_2/transformer_block_5/multi_head_attention_5/key/add�
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_2_transformer_block_5_multi_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
Fmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/EinsumEinsummodel_2/add_2/add:z:0]model_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2H
Fmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum�
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp�
<model_2/transformer_block_5/multi_head_attention_5/value/addAddV2Omodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum:output:0Smodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2>
<model_2/transformer_block_5/multi_head_attention_5/value/add�
8model_2/transformer_block_5/multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>2:
8model_2/transformer_block_5/multi_head_attention_5/Mul/y�
6model_2/transformer_block_5/multi_head_attention_5/MulMul@model_2/transformer_block_5/multi_head_attention_5/query/add:z:0Amodel_2/transformer_block_5/multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 28
6model_2/transformer_block_5/multi_head_attention_5/Mul�
@model_2/transformer_block_5/multi_head_attention_5/einsum/EinsumEinsum>model_2/transformer_block_5/multi_head_attention_5/key/add:z:0:model_2/transformer_block_5/multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2B
@model_2/transformer_block_5/multi_head_attention_5/einsum/Einsum�
Bmodel_2/transformer_block_5/multi_head_attention_5/softmax/SoftmaxSoftmaxImodel_2/transformer_block_5/multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2D
Bmodel_2/transformer_block_5/multi_head_attention_5/softmax/Softmax�
Cmodel_2/transformer_block_5/multi_head_attention_5/dropout/IdentityIdentityLmodel_2/transformer_block_5/multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������##2E
Cmodel_2/transformer_block_5/multi_head_attention_5/dropout/Identity�
Bmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/EinsumEinsumLmodel_2/transformer_block_5/multi_head_attention_5/dropout/Identity:output:0@model_2/transformer_block_5/multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2D
Bmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/Einsum�
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_2_transformer_block_5_multi_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
Qmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/EinsumEinsumKmodel_2/transformer_block_5/multi_head_attention_5/einsum_1/Einsum:output:0hmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe2S
Qmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum�
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOp_model_2_transformer_block_5_multi_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp�
Gmodel_2/transformer_block_5/multi_head_attention_5/attention_output/addAddV2Zmodel_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum:output:0^model_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2I
Gmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add�
/model_2/transformer_block_5/dropout_14/IdentityIdentityKmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:���������# 21
/model_2/transformer_block_5/dropout_14/Identity�
model_2/transformer_block_5/addAddV2model_2/add_2/add:z:08model_2/transformer_block_5/dropout_14/Identity:output:0*
T0*+
_output_shapes
:���������# 2!
model_2/transformer_block_5/add�
Qmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indices�
?model_2/transformer_block_5/layer_normalization_10/moments/meanMean#model_2/transformer_block_5/add:z:0Zmodel_2/transformer_block_5/layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2A
?model_2/transformer_block_5/layer_normalization_10/moments/mean�
Gmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradientStopGradientHmodel_2/transformer_block_5/layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2I
Gmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradient�
Lmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_5/add:z:0Pmodel_2/transformer_block_5/layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2N
Lmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifference�
Umodel_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indices�
Cmodel_2/transformer_block_5/layer_normalization_10/moments/varianceMeanPmodel_2/transformer_block_5/layer_normalization_10/moments/SquaredDifference:z:0^model_2/transformer_block_5/layer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2E
Cmodel_2/transformer_block_5/layer_normalization_10/moments/variance�
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/y�
@model_2/transformer_block_5/layer_normalization_10/batchnorm/addAddV2Lmodel_2/transformer_block_5/layer_normalization_10/moments/variance:output:0Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/add�
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/Rsqrt�
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_5_layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp�
@model_2/transformer_block_5/layer_normalization_10/batchnorm/mulMulFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/mul�
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1Mul#model_2/transformer_block_5/add:z:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1�
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2MulHmodel_2/transformer_block_5/layer_normalization_10/moments/mean:output:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2�
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp�
@model_2/transformer_block_5/layer_normalization_10/batchnorm/subSubSmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2B
@model_2/transformer_block_5/layer_normalization_10/batchnorm/sub�
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul_1:z:0Dmodel_2/transformer_block_5/layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2D
Bmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1�
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_5_sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp�
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/axes�
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/free�
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ShapeShapeFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Shape�
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis�
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2�
Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
Fmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Tmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1�
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const�
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/ProdProdMmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod�
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_1�
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1ProdOmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1�
Gmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis�
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concatConcatV2Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/free:output:0Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/axes:output:0Pmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat�
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stackPackImodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod:output:0Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stack�
Emodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose	TransposeFmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:0Kmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2G
Emodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose�
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReshapeReshapeImodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/transpose:y:0Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Reshape�
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMulMatMulLmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Reshape:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2D
Bmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMul�
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2�
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis�
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/Const_2:output:0Rmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1�
;model_2/transformer_block_5/sequential_5/dense_16/TensordotReshapeLmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/MatMul:product:0Mmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2=
;model_2/transformer_block_5/sequential_5/dense_16/Tensordot�
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_5_sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp�
9model_2/transformer_block_5/sequential_5/dense_16/BiasAddBiasAddDmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot:output:0Pmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2;
9model_2/transformer_block_5/sequential_5/dense_16/BiasAdd�
6model_2/transformer_block_5/sequential_5/dense_16/ReluReluBmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@28
6model_2/transformer_block_5/sequential_5/dense_16/Relu�
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_5_sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp�
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/axes�
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/free�
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ShapeShapeDmodel_2/transformer_block_5/sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Shape�
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis�
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Rmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2�
Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
Fmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Shape:output:0Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Tmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1�
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const�
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/ProdProdMmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod�
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_1�
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1ProdOmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1�
Gmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis�
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concatConcatV2Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/free:output:0Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/axes:output:0Pmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat�
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stackPackImodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod:output:0Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stack�
Emodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transpose	TransposeDmodel_2/transformer_block_5/sequential_5/dense_16/Relu:activations:0Kmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2G
Emodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transpose�
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReshapeReshapeImodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/transpose:y:0Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2E
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Reshape�
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMulMatMulLmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Reshape:output:0Rmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2D
Bmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMul�
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_2�
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis�
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/Const_2:output:0Rmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1�
;model_2/transformer_block_5/sequential_5/dense_17/TensordotReshapeLmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/MatMul:product:0Mmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2=
;model_2/transformer_block_5/sequential_5/dense_17/Tensordot�
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_5_sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp�
9model_2/transformer_block_5/sequential_5/dense_17/BiasAddBiasAddDmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot:output:0Pmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2;
9model_2/transformer_block_5/sequential_5/dense_17/BiasAdd�
/model_2/transformer_block_5/dropout_15/IdentityIdentityBmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:���������# 21
/model_2/transformer_block_5/dropout_15/Identity�
!model_2/transformer_block_5/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_10/batchnorm/add_1:z:08model_2/transformer_block_5/dropout_15/Identity:output:0*
T0*+
_output_shapes
:���������# 2#
!model_2/transformer_block_5/add_1�
Qmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indices�
?model_2/transformer_block_5/layer_normalization_11/moments/meanMean%model_2/transformer_block_5/add_1:z:0Zmodel_2/transformer_block_5/layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2A
?model_2/transformer_block_5/layer_normalization_11/moments/mean�
Gmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradientStopGradientHmodel_2/transformer_block_5/layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2I
Gmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradient�
Lmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_5/add_1:z:0Pmodel_2/transformer_block_5/layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2N
Lmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifference�
Umodel_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indices�
Cmodel_2/transformer_block_5/layer_normalization_11/moments/varianceMeanPmodel_2/transformer_block_5/layer_normalization_11/moments/SquaredDifference:z:0^model_2/transformer_block_5/layer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2E
Cmodel_2/transformer_block_5/layer_normalization_11/moments/variance�
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/y�
@model_2/transformer_block_5/layer_normalization_11/batchnorm/addAddV2Lmodel_2/transformer_block_5/layer_normalization_11/moments/variance:output:0Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/add�
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/RsqrtRsqrtDmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/Rsqrt�
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_2_transformer_block_5_layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp�
@model_2/transformer_block_5/layer_normalization_11/batchnorm/mulMulFmodel_2/transformer_block_5/layer_normalization_11/batchnorm/Rsqrt:y:0Wmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/mul�
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1Mul%model_2/transformer_block_5/add_1:z:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1�
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2MulHmodel_2/transformer_block_5/layer_normalization_11/moments/mean:output:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2�
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOpTmodel_2_transformer_block_5_layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp�
@model_2/transformer_block_5/layer_normalization_11/batchnorm/subSubSmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp:value:0Fmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2B
@model_2/transformer_block_5/layer_normalization_11/batchnorm/sub�
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1AddV2Fmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul_1:z:0Dmodel_2/transformer_block_5/layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2D
Bmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1�
model_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
model_2/flatten_2/Const�
model_2/flatten_2/ReshapeReshapeFmodel_2/transformer_block_5/layer_normalization_11/batchnorm/add_1:z:0 model_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:����������2
model_2/flatten_2/Reshape�
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/concatenate_2/concat/axis�
model_2/concatenate_2/concatConcatV2"model_2/flatten_2/Reshape:output:0input_6*model_2/concatenate_2/concat/axis:output:0*
N*
T0*(
_output_shapes
:����������2
model_2/concatenate_2/concat�
&model_2/dense_18/MatMul/ReadVariableOpReadVariableOp/model_2_dense_18_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02(
&model_2/dense_18/MatMul/ReadVariableOp�
model_2/dense_18/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
model_2/dense_18/MatMul�
'model_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_18/BiasAdd/ReadVariableOp�
model_2/dense_18/BiasAddBiasAdd!model_2/dense_18/MatMul:product:0/model_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
model_2/dense_18/BiasAdd�
model_2/dense_18/ReluRelu!model_2/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
model_2/dense_18/Relu�
model_2/dropout_16/IdentityIdentity#model_2/dense_18/Relu:activations:0*
T0*'
_output_shapes
:���������@2
model_2/dropout_16/Identity�
&model_2/dense_19/MatMul/ReadVariableOpReadVariableOp/model_2_dense_19_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_2/dense_19/MatMul/ReadVariableOp�
model_2/dense_19/MatMulMatMul$model_2/dropout_16/Identity:output:0.model_2/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
model_2/dense_19/MatMul�
'model_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_19/BiasAdd/ReadVariableOp�
model_2/dense_19/BiasAddBiasAdd!model_2/dense_19/MatMul:product:0/model_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
model_2/dense_19/BiasAdd�
model_2/dense_19/ReluRelu!model_2/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
model_2/dense_19/Relu�
model_2/dropout_17/IdentityIdentity#model_2/dense_19/Relu:activations:0*
T0*'
_output_shapes
:���������@2
model_2/dropout_17/Identity�
&model_2/dense_20/MatMul/ReadVariableOpReadVariableOp/model_2_dense_20_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_2/dense_20/MatMul/ReadVariableOp�
model_2/dense_20/MatMulMatMul$model_2/dropout_17/Identity:output:0.model_2/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_2/dense_20/MatMul�
'model_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_2/dense_20/BiasAdd/ReadVariableOp�
model_2/dense_20/BiasAddBiasAdd!model_2/dense_20/MatMul:product:0/model_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_2/dense_20/BiasAdd�
IdentityIdentity!model_2/dense_20/BiasAdd:output:07^model_2/batch_normalization_4/batchnorm/ReadVariableOp9^model_2/batch_normalization_4/batchnorm/ReadVariableOp_19^model_2/batch_normalization_4/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp7^model_2/batch_normalization_5/batchnorm/ReadVariableOp9^model_2/batch_normalization_5/batchnorm/ReadVariableOp_19^model_2/batch_normalization_5/batchnorm/ReadVariableOp_2;^model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp(^model_2/conv1d_4/BiasAdd/ReadVariableOp4^model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp(^model_2/conv1d_5/BiasAdd/ReadVariableOp4^model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp(^model_2/dense_18/BiasAdd/ReadVariableOp'^model_2/dense_18/MatMul/ReadVariableOp(^model_2/dense_19/BiasAdd/ReadVariableOp'^model_2/dense_19/MatMul/ReadVariableOp(^model_2/dense_20/BiasAdd/ReadVariableOp'^model_2/dense_20/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_2/embedding_4/embedding_lookupD^model_2/token_and_position_embedding_2/embedding_5/embedding_lookupL^model_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpP^model_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpL^model_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpP^model_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpW^model_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpa^model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpJ^model_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpT^model_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpV^model_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpL^model_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpV^model_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpI^model_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpK^model_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpI^model_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpK^model_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2p
6model_2/batch_normalization_4/batchnorm/ReadVariableOp6model_2/batch_normalization_4/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_18model_2/batch_normalization_4/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_4/batchnorm/ReadVariableOp_28model_2/batch_normalization_4/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_4/batchnorm/mul/ReadVariableOp2p
6model_2/batch_normalization_5/batchnorm/ReadVariableOp6model_2/batch_normalization_5/batchnorm/ReadVariableOp2t
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_18model_2/batch_normalization_5/batchnorm/ReadVariableOp_12t
8model_2/batch_normalization_5/batchnorm/ReadVariableOp_28model_2/batch_normalization_5/batchnorm/ReadVariableOp_22x
:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp:model_2/batch_normalization_5/batchnorm/mul/ReadVariableOp2R
'model_2/conv1d_4/BiasAdd/ReadVariableOp'model_2/conv1d_4/BiasAdd/ReadVariableOp2j
3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2R
'model_2/conv1d_5/BiasAdd/ReadVariableOp'model_2/conv1d_5/BiasAdd/ReadVariableOp2j
3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp3model_2/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2R
'model_2/dense_18/BiasAdd/ReadVariableOp'model_2/dense_18/BiasAdd/ReadVariableOp2P
&model_2/dense_18/MatMul/ReadVariableOp&model_2/dense_18/MatMul/ReadVariableOp2R
'model_2/dense_19/BiasAdd/ReadVariableOp'model_2/dense_19/BiasAdd/ReadVariableOp2P
&model_2/dense_19/MatMul/ReadVariableOp&model_2/dense_19/MatMul/ReadVariableOp2R
'model_2/dense_20/BiasAdd/ReadVariableOp'model_2/dense_20/BiasAdd/ReadVariableOp2P
&model_2/dense_20/MatMul/ReadVariableOp&model_2/dense_20/MatMul/ReadVariableOp2�
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup2�
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup2�
Kmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOpKmodel_2/transformer_block_5/layer_normalization_10/batchnorm/ReadVariableOp2�
Omodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOpOmodel_2/transformer_block_5/layer_normalization_10/batchnorm/mul/ReadVariableOp2�
Kmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOpKmodel_2/transformer_block_5/layer_normalization_11/batchnorm/ReadVariableOp2�
Omodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOpOmodel_2/transformer_block_5/layer_normalization_11/batchnorm/mul/ReadVariableOp2�
Vmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOpVmodel_2/transformer_block_5/multi_head_attention_5/attention_output/add/ReadVariableOp2�
`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp`model_2/transformer_block_5/multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2�
Imodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOpImodel_2/transformer_block_5/multi_head_attention_5/key/add/ReadVariableOp2�
Smodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOpSmodel_2/transformer_block_5/multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2�
Kmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOpKmodel_2/transformer_block_5/multi_head_attention_5/query/add/ReadVariableOp2�
Umodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_5/multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2�
Kmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOpKmodel_2/transformer_block_5/multi_head_attention_5/value/add/ReadVariableOp2�
Umodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOpUmodel_2/transformer_block_5/multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2�
Hmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOpHmodel_2/transformer_block_5/sequential_5/dense_16/BiasAdd/ReadVariableOp2�
Jmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOpJmodel_2/transformer_block_5/sequential_5/dense_16/Tensordot/ReadVariableOp2�
Hmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOpHmodel_2/transformer_block_5/sequential_5/dense_17/BiasAdd/ReadVariableOp2�
Jmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOpJmodel_2/transformer_block_5/sequential_5/dense_17/Tensordot/ReadVariableOp:Q M
(
_output_shapes
:����������R
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184756

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_185898

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:���������@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_187324

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1847232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_5_layer_call_fn_187570

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1853122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_187196
x'
#embedding_5_embedding_lookup_187183'
#embedding_4_embedding_lookup_187189
identity��embedding_4/embedding_lookup�embedding_5/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta�
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:���������2
range�
embedding_5/embedding_lookupResourceGather#embedding_5_embedding_lookup_187183range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_5/embedding_lookup/187183*'
_output_shapes
:��������� *
dtype02
embedding_5/embedding_lookup�
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_5/embedding_lookup/187183*'
_output_shapes
:��������� 2'
%embedding_5/embedding_lookup/Identity�
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:��������� 2)
'embedding_5/embedding_lookup/Identity_1q
embedding_4/CastCastx*

DstT0*

SrcT0*(
_output_shapes
:����������R2
embedding_4/Cast�
embedding_4/embedding_lookupResourceGather#embedding_4_embedding_lookup_187189embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_4/embedding_lookup/187189*,
_output_shapes
:����������R *
dtype02
embedding_4/embedding_lookup�
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_4/embedding_lookup/187189*,
_output_shapes
:����������R 2'
%embedding_4/embedding_lookup/Identity�
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������R 2)
'embedding_4/embedding_lookup/Identity_1�
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������R 2
add�
IdentityIdentityadd:z:0^embedding_4/embedding_lookup^embedding_5/embedding_lookup*
T0*,
_output_shapes
:����������R 2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������R::2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:K G
(
_output_shapes
:����������R

_user_specified_namex
�
m
A__inference_add_2_layer_call_and_return_conditional_losses_187589
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������# 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:���������# :���������# :U Q
+
_output_shapes
:���������# 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������# 
"
_user_specified_name
inputs/1
�[
�
C__inference_model_2_layer_call_and_return_conditional_losses_185938
input_5
input_6)
%token_and_position_embedding_2_185114)
%token_and_position_embedding_2_185116
conv1d_4_185146
conv1d_4_185148
conv1d_5_185179
conv1d_5_185181 
batch_normalization_4_185268 
batch_normalization_4_185270 
batch_normalization_4_185272 
batch_normalization_4_185274 
batch_normalization_5_185359 
batch_normalization_5_185361 
batch_normalization_5_185363 
batch_normalization_5_185365
transformer_block_5_185734
transformer_block_5_185736
transformer_block_5_185738
transformer_block_5_185740
transformer_block_5_185742
transformer_block_5_185744
transformer_block_5_185746
transformer_block_5_185748
transformer_block_5_185750
transformer_block_5_185752
transformer_block_5_185754
transformer_block_5_185756
transformer_block_5_185758
transformer_block_5_185760
transformer_block_5_185762
transformer_block_5_185764
dense_18_185819
dense_18_185821
dense_19_185876
dense_19_185878
dense_20_185932
dense_20_185934
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�"dropout_16/StatefulPartitionedCall�"dropout_17/StatefulPartitionedCall�6token_and_position_embedding_2/StatefulPartitionedCall�+transformer_block_5/StatefulPartitionedCall�
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_5%token_and_position_embedding_2_185114%token_and_position_embedding_2_185116*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_18510328
6token_and_position_embedding_2/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_185146conv1d_4_185148*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:����������R�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_1851352"
 conv1d_4/StatefulPartitionedCall�
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_1845912%
#average_pooling1d_6/PartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_185179conv1d_5_185181*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_1851682"
 conv1d_5/StatefulPartitionedCall�
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_1846212%
#average_pooling1d_8/PartitionedCall�
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_1846062%
#average_pooling1d_7/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_4_185268batch_normalization_4_185270batch_normalization_4_185272batch_normalization_4_185274*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1852212/
-batch_normalization_4/StatefulPartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_5_185359batch_normalization_5_185361batch_normalization_5_185363batch_normalization_5_185365*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1853122/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1853742
add_2/PartitionedCall�
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_185734transformer_block_5_185736transformer_block_5_185738transformer_block_5_185740transformer_block_5_185742transformer_block_5_185744transformer_block_5_185746transformer_block_5_185748transformer_block_5_185750transformer_block_5_185752transformer_block_5_185754transformer_block_5_185756transformer_block_5_185758transformer_block_5_185760transformer_block_5_185762transformer_block_5_185764*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_1855312-
+transformer_block_5/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1857732
flatten_2/PartitionedCall�
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0input_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_1857882
concatenate_2/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_185819dense_18_185821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_1858082"
 dense_18/StatefulPartitionedCall�
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_1858362$
"dropout_16/StatefulPartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_19_185876dense_19_185878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_1858652"
 dense_19/StatefulPartitionedCall�
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_1858932$
"dropout_17/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_20_185932dense_20_185934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1859212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:Q M
(
_output_shapes
:����������R
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�X
�
C__inference_model_2_layer_call_and_return_conditional_losses_186032
input_5
input_6)
%token_and_position_embedding_2_185942)
%token_and_position_embedding_2_185944
conv1d_4_185947
conv1d_4_185949
conv1d_5_185953
conv1d_5_185955 
batch_normalization_4_185960 
batch_normalization_4_185962 
batch_normalization_4_185964 
batch_normalization_4_185966 
batch_normalization_5_185969 
batch_normalization_5_185971 
batch_normalization_5_185973 
batch_normalization_5_185975
transformer_block_5_185979
transformer_block_5_185981
transformer_block_5_185983
transformer_block_5_185985
transformer_block_5_185987
transformer_block_5_185989
transformer_block_5_185991
transformer_block_5_185993
transformer_block_5_185995
transformer_block_5_185997
transformer_block_5_185999
transformer_block_5_186001
transformer_block_5_186003
transformer_block_5_186005
transformer_block_5_186007
transformer_block_5_186009
dense_18_186014
dense_18_186016
dense_19_186020
dense_19_186022
dense_20_186026
dense_20_186028
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�6token_and_position_embedding_2/StatefulPartitionedCall�+transformer_block_5/StatefulPartitionedCall�
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_5%token_and_position_embedding_2_185942%token_and_position_embedding_2_185944*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_18510328
6token_and_position_embedding_2/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_185947conv1d_4_185949*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:����������R�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_1851352"
 conv1d_4/StatefulPartitionedCall�
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_1845912%
#average_pooling1d_6/PartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_185953conv1d_5_185955*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_1851682"
 conv1d_5/StatefulPartitionedCall�
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_1846212%
#average_pooling1d_8/PartitionedCall�
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_1846062%
#average_pooling1d_7/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_4_185960batch_normalization_4_185962batch_normalization_4_185964batch_normalization_4_185966*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1852412/
-batch_normalization_4/StatefulPartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_5_185969batch_normalization_5_185971batch_normalization_5_185973batch_normalization_5_185975*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1853322/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1853742
add_2/PartitionedCall�
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_185979transformer_block_5_185981transformer_block_5_185983transformer_block_5_185985transformer_block_5_185987transformer_block_5_185989transformer_block_5_185991transformer_block_5_185993transformer_block_5_185995transformer_block_5_185997transformer_block_5_185999transformer_block_5_186001transformer_block_5_186003transformer_block_5_186005transformer_block_5_186007transformer_block_5_186009*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_1856582-
+transformer_block_5/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1857732
flatten_2/PartitionedCall�
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0input_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_1857882
concatenate_2/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_186014dense_18_186016*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_1858082"
 dense_18/StatefulPartitionedCall�
dropout_16/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_1858412
dropout_16/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_19_186020dense_19_186022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_1858652"
 dense_19/StatefulPartitionedCall�
dropout_17/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_1858982
dropout_17/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_20_186026dense_20_186028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1859212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:Q M
(
_output_shapes
:����������R
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�0
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_185312

inputs
assignmovingavg_185287
assignmovingavg_1_185293)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/185287*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_185287*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/185287*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/185287*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_185287AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/185287*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/185293*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_185293*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/185293*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/185293*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_185293AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/185293*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
~
)__inference_dense_20_layer_call_fn_188081

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1859212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_5_layer_call_fn_187488

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1848632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187475

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_185047
dense_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1850362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������# 
(
_user_specified_namedense_16_input
�
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_188000

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype0*

seedY2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
D__inference_dense_19_layer_call_and_return_conditional_losses_188026

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_dense_17_layer_call_and_return_conditional_losses_184988

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������#@
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_4_layer_call_fn_187419

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1852412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
~
)__inference_dense_19_layer_call_fn_188035

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_1858652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_185531

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity��/layer_normalization_10/batchnorm/ReadVariableOp�3layer_normalization_10/batchnorm/mul/ReadVariableOp�/layer_normalization_11/batchnorm/ReadVariableOp�3layer_normalization_11/batchnorm/mul/ReadVariableOp�:multi_head_attention_5/attention_output/add/ReadVariableOp�Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_5/key/add/ReadVariableOp�7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/query/add/ReadVariableOp�9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/value/add/ReadVariableOp�9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�,sequential_5/dense_16/BiasAdd/ReadVariableOp�.sequential_5/dense_16/Tensordot/ReadVariableOp�,sequential_5/dense_17/BiasAdd/ReadVariableOp�.sequential_5/dense_17/Tensordot/ReadVariableOp�
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/Einsum�
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOp�
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/query/add�
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/Einsum�
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOp�
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2 
multi_head_attention_5/key/add�
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/Einsum�
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOp�
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/value/add�
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>2
multi_head_attention_5/Mul/y�
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 2
multi_head_attention_5/Mul�
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/Einsum�
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2(
&multi_head_attention_5/softmax/Softmax�
,multi_head_attention_5/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,multi_head_attention_5/dropout/dropout/Const�
*multi_head_attention_5/dropout/dropout/MulMul0multi_head_attention_5/softmax/Softmax:softmax:05multi_head_attention_5/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������##2,
*multi_head_attention_5/dropout/dropout/Mul�
,multi_head_attention_5/dropout/dropout/ShapeShape0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_5/dropout/dropout/Shape�
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_5/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������##*
dtype0*

seedY2E
Cmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform�
5multi_head_attention_5/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_5/dropout/dropout/GreaterEqual/y�
3multi_head_attention_5/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_5/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_5/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������##25
3multi_head_attention_5/dropout/dropout/GreaterEqual�
+multi_head_attention_5/dropout/dropout/CastCast7multi_head_attention_5/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������##2-
+multi_head_attention_5/dropout/dropout/Cast�
,multi_head_attention_5/dropout/dropout/Mul_1Mul.multi_head_attention_5/dropout/dropout/Mul:z:0/multi_head_attention_5/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������##2.
,multi_head_attention_5/dropout/dropout/Mul_1�
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/dropout/Mul_1:z:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsum�
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp�
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2-
+multi_head_attention_5/attention_output/addy
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_14/dropout/Const�
dropout_14/dropout/MulMul/multi_head_attention_5/attention_output/add:z:0!dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:���������# 2
dropout_14/dropout/Mul�
dropout_14/dropout/ShapeShape/multi_head_attention_5/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape�
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:���������# *
dtype0*

seedY*
seed221
/dropout_14/dropout/random_uniform/RandomUniform�
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_14/dropout/GreaterEqual/y�
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������# 2!
dropout_14/dropout/GreaterEqual�
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������# 2
dropout_14/dropout/Cast�
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:���������# 2
dropout_14/dropout/Mul_1o
addAddV2inputsdropout_14/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������# 2
add�
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indices�
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_10/moments/mean�
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_10/moments/StopGradient�
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_10/moments/SquaredDifference�
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices�
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_10/moments/variance�
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_10/batchnorm/add/y�
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_10/batchnorm/add�
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_10/batchnorm/Rsqrt�
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOp�
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/mul�
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_1�
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_2�
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOp�
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/sub�
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/add_1�
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp�
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes�
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free�
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape�
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis�
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2�
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1�
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/Const�
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod�
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1�
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1�
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis�
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concat�
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stack�
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2+
)sequential_5/dense_16/Tensordot/transpose�
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_16/Tensordot/Reshape�
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2(
&sequential_5/dense_16/Tensordot/MatMul�
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2�
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis�
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1�
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2!
sequential_5/dense_16/Tensordot�
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp�
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/BiasAdd�
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/Relu�
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp�
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes�
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free�
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape�
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis�
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2�
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1�
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/Const�
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod�
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1�
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1�
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis�
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concat�
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stack�
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2+
)sequential_5/dense_17/Tensordot/transpose�
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_17/Tensordot/Reshape�
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&sequential_5/dense_17/Tensordot/MatMul�
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2�
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis�
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1�
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2!
sequential_5/dense_17/Tensordot�
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOp�
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2
sequential_5/dense_17/BiasAddy
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout_15/dropout/Const�
dropout_15/dropout/MulMul&sequential_5/dense_17/BiasAdd:output:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:���������# 2
dropout_15/dropout/Mul�
dropout_15/dropout/ShapeShape&sequential_5/dense_17/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape�
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:���������# *
dtype0*

seedY*
seed221
/dropout_15/dropout/random_uniform/RandomUniform�
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2#
!dropout_15/dropout/GreaterEqual/y�
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:���������# 2!
dropout_15/dropout/GreaterEqual�
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������# 2
dropout_15/dropout/Cast�
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:���������# 2
dropout_15/dropout/Mul_1�
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/dropout/Mul_1:z:0*
T0*+
_output_shapes
:���������# 2
add_1�
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indices�
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_11/moments/mean�
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_11/moments/StopGradient�
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_11/moments/SquaredDifference�
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices�
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_11/moments/variance�
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_11/batchnorm/add/y�
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_11/batchnorm/add�
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_11/batchnorm/Rsqrt�
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOp�
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/mul�
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_1�
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_2�
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOp�
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/sub�
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/add_1�
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_185005
dense_16_input
dense_16_184953
dense_16_184955
dense_17_184999
dense_17_185001
identity�� dense_16/StatefulPartitionedCall� dense_17/StatefulPartitionedCall�
 dense_16/StatefulPartitionedCallStatefulPartitionedCalldense_16_inputdense_16_184953dense_16_184955*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������#@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1849422"
 dense_16/StatefulPartitionedCall�
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_184999dense_17_185001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1849882"
 dense_17/StatefulPartitionedCall�
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:[ W
+
_output_shapes
:���������# 
(
_user_specified_namedense_16_input
�0
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187291

inputs
assignmovingavg_187266
assignmovingavg_1_187272)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187266*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_187266*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187266*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/187266*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_187266AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/187266*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187272*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_187272*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187272*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/187272*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_187272AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/187272*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs
�
~
)__inference_conv1d_5_layer_call_fn_187255

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_1851682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������� 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
~
)__inference_dense_18_layer_call_fn_187988

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_1858082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�J
�
H__inference_sequential_5_layer_call_and_return_conditional_losses_188138

inputs.
*dense_16_tensordot_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource.
*dense_17_tensordot_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity��dense_16/BiasAdd/ReadVariableOp�!dense_16/Tensordot/ReadVariableOp�dense_17/BiasAdd/ReadVariableOp�!dense_17/Tensordot/ReadVariableOp�
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_16/Tensordot/ReadVariableOp|
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_16/Tensordot/axes�
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_16/Tensordot/freej
dense_16/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_16/Tensordot/Shape�
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/GatherV2/axis�
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2�
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_16/Tensordot/GatherV2_1/axis�
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_16/Tensordot/GatherV2_1~
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const�
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod�
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_16/Tensordot/Const_1�
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_16/Tensordot/Prod_1�
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_16/Tensordot/concat/axis�
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat�
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/stack�
dense_16/Tensordot/transpose	Transposeinputs"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2
dense_16/Tensordot/transpose�
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_16/Tensordot/Reshape�
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_16/Tensordot/MatMul�
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_16/Tensordot/Const_2�
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_16/Tensordot/concat_1/axis�
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_16/Tensordot/concat_1�
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2
dense_16/Tensordot�
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp�
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2
dense_16/BiasAddw
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
dense_16/Relu�
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_17/Tensordot/ReadVariableOp|
dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_17/Tensordot/axes�
dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_17/Tensordot/free
dense_17/Tensordot/ShapeShapedense_16/Relu:activations:0*
T0*
_output_shapes
:2
dense_17/Tensordot/Shape�
 dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/GatherV2/axis�
dense_17/Tensordot/GatherV2GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/free:output:0)dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2�
"dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_17/Tensordot/GatherV2_1/axis�
dense_17/Tensordot/GatherV2_1GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/axes:output:0+dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_17/Tensordot/GatherV2_1~
dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const�
dense_17/Tensordot/ProdProd$dense_17/Tensordot/GatherV2:output:0!dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod�
dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_1�
dense_17/Tensordot/Prod_1Prod&dense_17/Tensordot/GatherV2_1:output:0#dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_17/Tensordot/Prod_1�
dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_17/Tensordot/concat/axis�
dense_17/Tensordot/concatConcatV2 dense_17/Tensordot/free:output:0 dense_17/Tensordot/axes:output:0'dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat�
dense_17/Tensordot/stackPack dense_17/Tensordot/Prod:output:0"dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/stack�
dense_17/Tensordot/transpose	Transposedense_16/Relu:activations:0"dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2
dense_17/Tensordot/transpose�
dense_17/Tensordot/ReshapeReshape dense_17/Tensordot/transpose:y:0!dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_17/Tensordot/Reshape�
dense_17/Tensordot/MatMulMatMul#dense_17/Tensordot/Reshape:output:0)dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_17/Tensordot/MatMul�
dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_17/Tensordot/Const_2�
 dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_17/Tensordot/concat_1/axis�
dense_17/Tensordot/concat_1ConcatV2$dense_17/Tensordot/GatherV2:output:0#dense_17/Tensordot/Const_2:output:0)dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_17/Tensordot/concat_1�
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2
dense_17/Tensordot�
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_17/BiasAdd/ReadVariableOp�
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2
dense_17/BiasAdd�
IdentityIdentitydense_17/BiasAdd:output:0 ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
-__inference_sequential_5_layer_call_fn_185074
dense_16_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_16_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1850632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������# ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:���������# 
(
_user_specified_namedense_16_input
��
�
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_187870

inputsF
Bmulti_head_attention_5_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_query_add_readvariableop_resourceD
@multi_head_attention_5_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_5_key_add_readvariableop_resourceF
Bmulti_head_attention_5_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_5_value_add_readvariableop_resourceQ
Mmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_5_attention_output_add_readvariableop_resource@
<layer_normalization_10_batchnorm_mul_readvariableop_resource<
8layer_normalization_10_batchnorm_readvariableop_resource;
7sequential_5_dense_16_tensordot_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource;
7sequential_5_dense_17_tensordot_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource@
<layer_normalization_11_batchnorm_mul_readvariableop_resource<
8layer_normalization_11_batchnorm_readvariableop_resource
identity��/layer_normalization_10/batchnorm/ReadVariableOp�3layer_normalization_10/batchnorm/mul/ReadVariableOp�/layer_normalization_11/batchnorm/ReadVariableOp�3layer_normalization_11/batchnorm/mul/ReadVariableOp�:multi_head_attention_5/attention_output/add/ReadVariableOp�Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_5/key/add/ReadVariableOp�7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/query/add/ReadVariableOp�9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_5/value/add/ReadVariableOp�9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�,sequential_5/dense_16/BiasAdd/ReadVariableOp�.sequential_5/dense_16/Tensordot/ReadVariableOp�,sequential_5/dense_17/BiasAdd/ReadVariableOp�.sequential_5/dense_17/Tensordot/ReadVariableOp�
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/query/einsum/EinsumEinsuminputsAmulti_head_attention_5/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/query/einsum/Einsum�
/multi_head_attention_5/query/add/ReadVariableOpReadVariableOp8multi_head_attention_5_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/query/add/ReadVariableOp�
 multi_head_attention_5/query/addAddV23multi_head_attention_5/query/einsum/Einsum:output:07multi_head_attention_5/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/query/add�
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_5_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp�
(multi_head_attention_5/key/einsum/EinsumEinsuminputs?multi_head_attention_5/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2*
(multi_head_attention_5/key/einsum/Einsum�
-multi_head_attention_5/key/add/ReadVariableOpReadVariableOp6multi_head_attention_5_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_5/key/add/ReadVariableOp�
multi_head_attention_5/key/addAddV21multi_head_attention_5/key/einsum/Einsum:output:05multi_head_attention_5/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2 
multi_head_attention_5/key/add�
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_5_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp�
*multi_head_attention_5/value/einsum/EinsumEinsuminputsAmulti_head_attention_5/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������# *
equationabc,cde->abde2,
*multi_head_attention_5/value/einsum/Einsum�
/multi_head_attention_5/value/add/ReadVariableOpReadVariableOp8multi_head_attention_5_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_5/value/add/ReadVariableOp�
 multi_head_attention_5/value/addAddV23multi_head_attention_5/value/einsum/Einsum:output:07multi_head_attention_5/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������# 2"
 multi_head_attention_5/value/add�
multi_head_attention_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *�5>2
multi_head_attention_5/Mul/y�
multi_head_attention_5/MulMul$multi_head_attention_5/query/add:z:0%multi_head_attention_5/Mul/y:output:0*
T0*/
_output_shapes
:���������# 2
multi_head_attention_5/Mul�
$multi_head_attention_5/einsum/EinsumEinsum"multi_head_attention_5/key/add:z:0multi_head_attention_5/Mul:z:0*
N*
T0*/
_output_shapes
:���������##*
equationaecd,abcd->acbe2&
$multi_head_attention_5/einsum/Einsum�
&multi_head_attention_5/softmax/SoftmaxSoftmax-multi_head_attention_5/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������##2(
&multi_head_attention_5/softmax/Softmax�
'multi_head_attention_5/dropout/IdentityIdentity0multi_head_attention_5/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������##2)
'multi_head_attention_5/dropout/Identity�
&multi_head_attention_5/einsum_1/EinsumEinsum0multi_head_attention_5/dropout/Identity:output:0$multi_head_attention_5/value/add:z:0*
N*
T0*/
_output_shapes
:���������# *
equationacbe,aecd->abcd2(
&multi_head_attention_5/einsum_1/Einsum�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_5_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp�
5multi_head_attention_5/attention_output/einsum/EinsumEinsum/multi_head_attention_5/einsum_1/Einsum:output:0Lmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������# *
equationabcd,cde->abe27
5multi_head_attention_5/attention_output/einsum/Einsum�
:multi_head_attention_5/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_5_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_5/attention_output/add/ReadVariableOp�
+multi_head_attention_5/attention_output/addAddV2>multi_head_attention_5/attention_output/einsum/Einsum:output:0Bmulti_head_attention_5/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2-
+multi_head_attention_5/attention_output/add�
dropout_14/IdentityIdentity/multi_head_attention_5/attention_output/add:z:0*
T0*+
_output_shapes
:���������# 2
dropout_14/Identityo
addAddV2inputsdropout_14/Identity:output:0*
T0*+
_output_shapes
:���������# 2
add�
5layer_normalization_10/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_10/moments/mean/reduction_indices�
#layer_normalization_10/moments/meanMeanadd:z:0>layer_normalization_10/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_10/moments/mean�
+layer_normalization_10/moments/StopGradientStopGradient,layer_normalization_10/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_10/moments/StopGradient�
0layer_normalization_10/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_10/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_10/moments/SquaredDifference�
9layer_normalization_10/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_10/moments/variance/reduction_indices�
'layer_normalization_10/moments/varianceMean4layer_normalization_10/moments/SquaredDifference:z:0Blayer_normalization_10/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_10/moments/variance�
&layer_normalization_10/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_10/batchnorm/add/y�
$layer_normalization_10/batchnorm/addAddV20layer_normalization_10/moments/variance:output:0/layer_normalization_10/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_10/batchnorm/add�
&layer_normalization_10/batchnorm/RsqrtRsqrt(layer_normalization_10/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_10/batchnorm/Rsqrt�
3layer_normalization_10/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_10_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_10/batchnorm/mul/ReadVariableOp�
$layer_normalization_10/batchnorm/mulMul*layer_normalization_10/batchnorm/Rsqrt:y:0;layer_normalization_10/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/mul�
&layer_normalization_10/batchnorm/mul_1Muladd:z:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_1�
&layer_normalization_10/batchnorm/mul_2Mul,layer_normalization_10/moments/mean:output:0(layer_normalization_10/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/mul_2�
/layer_normalization_10/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_10_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_10/batchnorm/ReadVariableOp�
$layer_normalization_10/batchnorm/subSub7layer_normalization_10/batchnorm/ReadVariableOp:value:0*layer_normalization_10/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_10/batchnorm/sub�
&layer_normalization_10/batchnorm/add_1AddV2*layer_normalization_10/batchnorm/mul_1:z:0(layer_normalization_10/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_10/batchnorm/add_1�
.sequential_5/dense_16/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_16_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_5/dense_16/Tensordot/ReadVariableOp�
$sequential_5/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_16/Tensordot/axes�
$sequential_5/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_16/Tensordot/free�
%sequential_5/dense_16/Tensordot/ShapeShape*layer_normalization_10/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/Shape�
-sequential_5/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/GatherV2/axis�
(sequential_5/dense_16/Tensordot/GatherV2GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/free:output:06sequential_5/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/GatherV2�
/sequential_5/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_16/Tensordot/GatherV2_1/axis�
*sequential_5/dense_16/Tensordot/GatherV2_1GatherV2.sequential_5/dense_16/Tensordot/Shape:output:0-sequential_5/dense_16/Tensordot/axes:output:08sequential_5/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_16/Tensordot/GatherV2_1�
%sequential_5/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_16/Tensordot/Const�
$sequential_5/dense_16/Tensordot/ProdProd1sequential_5/dense_16/Tensordot/GatherV2:output:0.sequential_5/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_16/Tensordot/Prod�
'sequential_5/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_16/Tensordot/Const_1�
&sequential_5/dense_16/Tensordot/Prod_1Prod3sequential_5/dense_16/Tensordot/GatherV2_1:output:00sequential_5/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_16/Tensordot/Prod_1�
+sequential_5/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_16/Tensordot/concat/axis�
&sequential_5/dense_16/Tensordot/concatConcatV2-sequential_5/dense_16/Tensordot/free:output:0-sequential_5/dense_16/Tensordot/axes:output:04sequential_5/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_16/Tensordot/concat�
%sequential_5/dense_16/Tensordot/stackPack-sequential_5/dense_16/Tensordot/Prod:output:0/sequential_5/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_16/Tensordot/stack�
)sequential_5/dense_16/Tensordot/transpose	Transpose*layer_normalization_10/batchnorm/add_1:z:0/sequential_5/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2+
)sequential_5/dense_16/Tensordot/transpose�
'sequential_5/dense_16/Tensordot/ReshapeReshape-sequential_5/dense_16/Tensordot/transpose:y:0.sequential_5/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_16/Tensordot/Reshape�
&sequential_5/dense_16/Tensordot/MatMulMatMul0sequential_5/dense_16/Tensordot/Reshape:output:06sequential_5/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2(
&sequential_5/dense_16/Tensordot/MatMul�
'sequential_5/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_5/dense_16/Tensordot/Const_2�
-sequential_5/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_16/Tensordot/concat_1/axis�
(sequential_5/dense_16/Tensordot/concat_1ConcatV21sequential_5/dense_16/Tensordot/GatherV2:output:00sequential_5/dense_16/Tensordot/Const_2:output:06sequential_5/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_16/Tensordot/concat_1�
sequential_5/dense_16/TensordotReshape0sequential_5/dense_16/Tensordot/MatMul:product:01sequential_5/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2!
sequential_5/dense_16/Tensordot�
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOp�
sequential_5/dense_16/BiasAddBiasAdd(sequential_5/dense_16/Tensordot:output:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/BiasAdd�
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
sequential_5/dense_16/Relu�
.sequential_5/dense_17/Tensordot/ReadVariableOpReadVariableOp7sequential_5_dense_17_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_5/dense_17/Tensordot/ReadVariableOp�
$sequential_5/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_5/dense_17/Tensordot/axes�
$sequential_5/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_5/dense_17/Tensordot/free�
%sequential_5/dense_17/Tensordot/ShapeShape(sequential_5/dense_16/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/Shape�
-sequential_5/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/GatherV2/axis�
(sequential_5/dense_17/Tensordot/GatherV2GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/free:output:06sequential_5/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/GatherV2�
/sequential_5/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_5/dense_17/Tensordot/GatherV2_1/axis�
*sequential_5/dense_17/Tensordot/GatherV2_1GatherV2.sequential_5/dense_17/Tensordot/Shape:output:0-sequential_5/dense_17/Tensordot/axes:output:08sequential_5/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_5/dense_17/Tensordot/GatherV2_1�
%sequential_5/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_5/dense_17/Tensordot/Const�
$sequential_5/dense_17/Tensordot/ProdProd1sequential_5/dense_17/Tensordot/GatherV2:output:0.sequential_5/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_5/dense_17/Tensordot/Prod�
'sequential_5/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_1�
&sequential_5/dense_17/Tensordot/Prod_1Prod3sequential_5/dense_17/Tensordot/GatherV2_1:output:00sequential_5/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_5/dense_17/Tensordot/Prod_1�
+sequential_5/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_5/dense_17/Tensordot/concat/axis�
&sequential_5/dense_17/Tensordot/concatConcatV2-sequential_5/dense_17/Tensordot/free:output:0-sequential_5/dense_17/Tensordot/axes:output:04sequential_5/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_5/dense_17/Tensordot/concat�
%sequential_5/dense_17/Tensordot/stackPack-sequential_5/dense_17/Tensordot/Prod:output:0/sequential_5/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_17/Tensordot/stack�
)sequential_5/dense_17/Tensordot/transpose	Transpose(sequential_5/dense_16/Relu:activations:0/sequential_5/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������#@2+
)sequential_5/dense_17/Tensordot/transpose�
'sequential_5/dense_17/Tensordot/ReshapeReshape-sequential_5/dense_17/Tensordot/transpose:y:0.sequential_5/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2)
'sequential_5/dense_17/Tensordot/Reshape�
&sequential_5/dense_17/Tensordot/MatMulMatMul0sequential_5/dense_17/Tensordot/Reshape:output:06sequential_5/dense_17/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2(
&sequential_5/dense_17/Tensordot/MatMul�
'sequential_5/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_5/dense_17/Tensordot/Const_2�
-sequential_5/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_5/dense_17/Tensordot/concat_1/axis�
(sequential_5/dense_17/Tensordot/concat_1ConcatV21sequential_5/dense_17/Tensordot/GatherV2:output:00sequential_5/dense_17/Tensordot/Const_2:output:06sequential_5/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_5/dense_17/Tensordot/concat_1�
sequential_5/dense_17/TensordotReshape0sequential_5/dense_17/Tensordot/MatMul:product:01sequential_5/dense_17/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������# 2!
sequential_5/dense_17/Tensordot�
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOp�
sequential_5/dense_17/BiasAddBiasAdd(sequential_5/dense_17/Tensordot:output:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2
sequential_5/dense_17/BiasAdd�
dropout_15/IdentityIdentity&sequential_5/dense_17/BiasAdd:output:0*
T0*+
_output_shapes
:���������# 2
dropout_15/Identity�
add_1AddV2*layer_normalization_10/batchnorm/add_1:z:0dropout_15/Identity:output:0*
T0*+
_output_shapes
:���������# 2
add_1�
5layer_normalization_11/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_11/moments/mean/reduction_indices�
#layer_normalization_11/moments/meanMean	add_1:z:0>layer_normalization_11/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2%
#layer_normalization_11/moments/mean�
+layer_normalization_11/moments/StopGradientStopGradient,layer_normalization_11/moments/mean:output:0*
T0*+
_output_shapes
:���������#2-
+layer_normalization_11/moments/StopGradient�
0layer_normalization_11/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_11/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������# 22
0layer_normalization_11/moments/SquaredDifference�
9layer_normalization_11/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_11/moments/variance/reduction_indices�
'layer_normalization_11/moments/varianceMean4layer_normalization_11/moments/SquaredDifference:z:0Blayer_normalization_11/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������#*
	keep_dims(2)
'layer_normalization_11/moments/variance�
&layer_normalization_11/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52(
&layer_normalization_11/batchnorm/add/y�
$layer_normalization_11/batchnorm/addAddV20layer_normalization_11/moments/variance:output:0/layer_normalization_11/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������#2&
$layer_normalization_11/batchnorm/add�
&layer_normalization_11/batchnorm/RsqrtRsqrt(layer_normalization_11/batchnorm/add:z:0*
T0*+
_output_shapes
:���������#2(
&layer_normalization_11/batchnorm/Rsqrt�
3layer_normalization_11/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_11_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_11/batchnorm/mul/ReadVariableOp�
$layer_normalization_11/batchnorm/mulMul*layer_normalization_11/batchnorm/Rsqrt:y:0;layer_normalization_11/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/mul�
&layer_normalization_11/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_1�
&layer_normalization_11/batchnorm/mul_2Mul,layer_normalization_11/moments/mean:output:0(layer_normalization_11/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/mul_2�
/layer_normalization_11/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_11_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_11/batchnorm/ReadVariableOp�
$layer_normalization_11/batchnorm/subSub7layer_normalization_11/batchnorm/ReadVariableOp:value:0*layer_normalization_11/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������# 2&
$layer_normalization_11/batchnorm/sub�
&layer_normalization_11/batchnorm/add_1AddV2*layer_normalization_11/batchnorm/mul_1:z:0(layer_normalization_11/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������# 2(
&layer_normalization_11/batchnorm/add_1�
IdentityIdentity*layer_normalization_11/batchnorm/add_1:z:00^layer_normalization_10/batchnorm/ReadVariableOp4^layer_normalization_10/batchnorm/mul/ReadVariableOp0^layer_normalization_11/batchnorm/ReadVariableOp4^layer_normalization_11/batchnorm/mul/ReadVariableOp;^multi_head_attention_5/attention_output/add/ReadVariableOpE^multi_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_5/key/add/ReadVariableOp8^multi_head_attention_5/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/query/add/ReadVariableOp:^multi_head_attention_5/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_5/value/add/ReadVariableOp:^multi_head_attention_5/value/einsum/Einsum/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp/^sequential_5/dense_16/Tensordot/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp/^sequential_5/dense_17/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:���������# ::::::::::::::::2b
/layer_normalization_10/batchnorm/ReadVariableOp/layer_normalization_10/batchnorm/ReadVariableOp2j
3layer_normalization_10/batchnorm/mul/ReadVariableOp3layer_normalization_10/batchnorm/mul/ReadVariableOp2b
/layer_normalization_11/batchnorm/ReadVariableOp/layer_normalization_11/batchnorm/ReadVariableOp2j
3layer_normalization_11/batchnorm/mul/ReadVariableOp3layer_normalization_11/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_5/attention_output/add/ReadVariableOp:multi_head_attention_5/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_5/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_5/key/add/ReadVariableOp-multi_head_attention_5/key/add/ReadVariableOp2r
7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp7multi_head_attention_5/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/query/add/ReadVariableOp/multi_head_attention_5/query/add/ReadVariableOp2v
9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp9multi_head_attention_5/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_5/value/add/ReadVariableOp/multi_head_attention_5/value/add/ReadVariableOp2v
9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp9multi_head_attention_5/value/einsum/Einsum/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2`
.sequential_5/dense_16/Tensordot/ReadVariableOp.sequential_5/dense_16/Tensordot/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2`
.sequential_5/dense_17/Tensordot/ReadVariableOp.sequential_5/dense_17/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
~
)__inference_dense_17_layer_call_fn_188300

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1849882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������# 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������#@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������#@
 
_user_specified_nameinputs
�
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_187950

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������# :S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_186463
input_5
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5input_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_1845822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������R
!
_user_specified_name	input_5:PL
'
_output_shapes
:���������
!
_user_specified_name	input_6
�X
�
C__inference_model_2_layer_call_and_return_conditional_losses_186302

inputs
inputs_1)
%token_and_position_embedding_2_186212)
%token_and_position_embedding_2_186214
conv1d_4_186217
conv1d_4_186219
conv1d_5_186223
conv1d_5_186225 
batch_normalization_4_186230 
batch_normalization_4_186232 
batch_normalization_4_186234 
batch_normalization_4_186236 
batch_normalization_5_186239 
batch_normalization_5_186241 
batch_normalization_5_186243 
batch_normalization_5_186245
transformer_block_5_186249
transformer_block_5_186251
transformer_block_5_186253
transformer_block_5_186255
transformer_block_5_186257
transformer_block_5_186259
transformer_block_5_186261
transformer_block_5_186263
transformer_block_5_186265
transformer_block_5_186267
transformer_block_5_186269
transformer_block_5_186271
transformer_block_5_186273
transformer_block_5_186275
transformer_block_5_186277
transformer_block_5_186279
dense_18_186284
dense_18_186286
dense_19_186290
dense_19_186292
dense_20_186296
dense_20_186298
identity��-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall� conv1d_4/StatefulPartitionedCall� conv1d_5/StatefulPartitionedCall� dense_18/StatefulPartitionedCall� dense_19/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�6token_and_position_embedding_2/StatefulPartitionedCall�+transformer_block_5/StatefulPartitionedCall�
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_2_186212%token_and_position_embedding_2_186214*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������R *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *c
f^R\
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_18510328
6token_and_position_embedding_2/StatefulPartitionedCall�
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0conv1d_4_186217conv1d_4_186219*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:����������R�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_1851352"
 conv1d_4/StatefulPartitionedCall�
#average_pooling1d_6/PartitionedCallPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_1845912%
#average_pooling1d_6/PartitionedCall�
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_6/PartitionedCall:output:0conv1d_5_186223conv1d_5_186225*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_1851682"
 conv1d_5/StatefulPartitionedCall�
#average_pooling1d_8/PartitionedCallPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_1846212%
#average_pooling1d_8/PartitionedCall�
#average_pooling1d_7/PartitionedCallPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_1846062%
#average_pooling1d_7/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_7/PartitionedCall:output:0batch_normalization_4_186230batch_normalization_4_186232batch_normalization_4_186234batch_normalization_4_186236*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1852412/
-batch_normalization_4/StatefulPartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_8/PartitionedCall:output:0batch_normalization_5_186239batch_normalization_5_186241batch_normalization_5_186243batch_normalization_5_186245*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1853322/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:06batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_1853742
add_2/PartitionedCall�
+transformer_block_5/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0transformer_block_5_186249transformer_block_5_186251transformer_block_5_186253transformer_block_5_186255transformer_block_5_186257transformer_block_5_186259transformer_block_5_186261transformer_block_5_186263transformer_block_5_186265transformer_block_5_186267transformer_block_5_186269transformer_block_5_186271transformer_block_5_186273transformer_block_5_186275transformer_block_5_186277transformer_block_5_186279*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������# *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_1856582-
+transformer_block_5/StatefulPartitionedCall�
flatten_2/PartitionedCallPartitionedCall4transformer_block_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_1857732
flatten_2/PartitionedCall�
concatenate_2/PartitionedCallPartitionedCall"flatten_2/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_2_layer_call_and_return_conditional_losses_1857882
concatenate_2/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_18_186284dense_18_186286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_18_layer_call_and_return_conditional_losses_1858082"
 dense_18/StatefulPartitionedCall�
dropout_16/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_1858412
dropout_16/PartitionedCall�
 dense_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_19_186290dense_19_186292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_1858652"
 dense_19/StatefulPartitionedCall�
dropout_17/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_1858982
dropout_17/PartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_20_186296dense_20_186298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_1859212"
 dense_20/StatefulPartitionedCall�
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall!^dense_19/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_5/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:����������R:���������::::::::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_5/StatefulPartitionedCall+transformer_block_5/StatefulPartitionedCall:P L
(
_output_shapes
:����������R
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
� 
�
D__inference_dense_16_layer_call_and_return_conditional_losses_188252

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������# 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������#@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������#@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������#@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������#@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������# ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������# 
 
_user_specified_nameinputs
�
�
6__inference_batch_normalization_5_layer_call_fn_187501

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1848962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:������������������ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������ 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_51
serving_default_input_5:0����������R
;
input_60
serving_default_input_6:0���������<
dense_200
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�G
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer_with_weights-7
layer-16
layer-17
layer_with_weights-8
layer-18
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�__call__
�_default_save_signature"�B
_tf_keras_network�A{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_6", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["average_pooling1d_6", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_7", "inbound_nodes": [[["conv1d_5", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_8", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_8", "inbound_nodes": [[["token_and_position_embedding_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["average_pooling1d_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["average_pooling1d_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}], ["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_5", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["transformer_block_5", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["flatten_2", 0, 0, {}], ["input_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_18", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_16", "inbound_nodes": [[["dense_18", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_19", "inbound_nodes": [[["dropout_16", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_17", "inbound_nodes": [[["dense_19", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["dropout_17", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0], ["input_6", 0, 0]], "output_layers": [["dense_20", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 10500]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10500]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.00010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10500]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
�
	token_emb
pos_emb
	variables
trainable_variables
regularization_losses
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�	

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500, 32]}}
�
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling1D", "name": "average_pooling1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_6", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 350, 128]}}
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling1D", "name": "average_pooling1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_7", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "AveragePooling1D", "name": "average_pooling1d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_8", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=	variables
>trainable_variables
?regularization_losses
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 35, 32]}, {"class_name": "TensorShape", "items": [null, 35, 32]}]}
�
Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TransformerBlock", "name": "transformer_block_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1120]}, {"class_name": "TensorShape", "items": [null, 8]}]}
�

`kernel
abias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_18", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1128]}}
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_19", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
�

tkernel
ubias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�
	zdecay
{learning_rate
|momentum
}iter momentum�!momentum�*momentum�+momentum�9momentum�:momentum�Bmomentum�Cmomentum�`momentum�amomentum�jmomentum�kmomentum�tmomentum�umomentum�~momentum�momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum��momentum�"
	optimizer
�
~0
1
 2
!3
*4
+5
96
:7
;8
<9
B10
C11
D12
E13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
`30
a31
j32
k33
t34
u35"
trackable_list_wrapper
�
~0
1
 2
!3
*4
+5
96
:7
B8
C9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
`26
a27
j28
k29
t30
u31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�
~
embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10500]}}
�

embeddings
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10500, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$ �2conv1d_4/kernel
:�2conv1d_4/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
"	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
#trainable_variables
$regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
&	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
'trainable_variables
(regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$	� 2conv1d_5/kernel
: 2conv1d_5/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
,	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
-trainable_variables
.regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
1trainable_variables
2regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
4	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
5trainable_variables
6regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
=	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
>trainable_variables
?regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_5/gamma
(:& 2batch_normalization_5/beta
1:/  (2!batch_normalization_5/moving_mean
5:3  (2%batch_normalization_5/moving_variance
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
F	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Gtrainable_variables
Hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
J	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Ktrainable_variables
Lregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "MultiHeadAttention", "name": "multi_head_attention_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_5", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
�
�layer_with_weights-0
�layer-0
�layer_with_weights-1
�layer-1
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_sequential�{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_16_input"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_16_input"}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LayerNormalization", "name": "layer_normalization_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_10", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
	�axis

�gamma
	�beta
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LayerNormalization", "name": "layer_normalization_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_11", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_14", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_15", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
T	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Utrainable_variables
Vregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
X	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
Ytrainable_variables
Zregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
]trainable_variables
^regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�@2dense_18/kernel
:@2dense_18/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
b	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
ctrainable_variables
dregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
f	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
gtrainable_variables
hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_19/kernel
:@2dense_19/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
l	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
mtrainable_variables
nregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
p	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
qtrainable_variables
rregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_20/kernel
:2dense_20/bias
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
v	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
wtrainable_variables
xregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_2/embedding_4/embeddings
H:F	�R 25token_and_position_embedding_2/embedding_5/embeddings
M:K  27transformer_block_5/multi_head_attention_5/query/kernel
G:E 25transformer_block_5/multi_head_attention_5/query/bias
K:I  25transformer_block_5/multi_head_attention_5/key/kernel
E:C 23transformer_block_5/multi_head_attention_5/key/bias
M:K  27transformer_block_5/multi_head_attention_5/value/kernel
G:E 25transformer_block_5/multi_head_attention_5/value/bias
X:V  2Btransformer_block_5/multi_head_attention_5/attention_output/kernel
N:L 2@transformer_block_5/multi_head_attention_5/attention_output/bias
!: @2dense_16/kernel
:@2dense_16/bias
!:@ 2dense_17/kernel
: 2dense_17/bias
>:< 20transformer_block_5/layer_normalization_10/gamma
=:; 2/transformer_block_5/layer_normalization_10/beta
>:< 20transformer_block_5/layer_normalization_11/gamma
=:; 2/transformer_block_5/layer_normalization_11/beta
(
�0"
trackable_list_wrapper
<
;0
<1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 1, 32]}}
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 32]}}
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 64]}}
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
N0
O1
P2
Q3
R4
S5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�	variables
�metrics
�non_trainable_variables
 �layer_regularization_losses
�layer_metrics
�layers
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
1:/ �2SGD/conv1d_4/kernel/momentum
':%�2SGD/conv1d_4/bias/momentum
1:/	� 2SGD/conv1d_5/kernel/momentum
&:$ 2SGD/conv1d_5/bias/momentum
4:2 2(SGD/batch_normalization_4/gamma/momentum
3:1 2'SGD/batch_normalization_4/beta/momentum
4:2 2(SGD/batch_normalization_5/gamma/momentum
3:1 2'SGD/batch_normalization_5/beta/momentum
-:+	�@2SGD/dense_18/kernel/momentum
&:$@2SGD/dense_18/bias/momentum
,:*@@2SGD/dense_19/kernel/momentum
&:$@2SGD/dense_19/bias/momentum
,:*@2SGD/dense_20/kernel/momentum
&:$2SGD/dense_20/bias/momentum
R:P 2BSGD/token_and_position_embedding_2/embedding_4/embeddings/momentum
S:Q	�R 2BSGD/token_and_position_embedding_2/embedding_5/embeddings/momentum
X:V  2DSGD/transformer_block_5/multi_head_attention_5/query/kernel/momentum
R:P 2BSGD/transformer_block_5/multi_head_attention_5/query/bias/momentum
V:T  2BSGD/transformer_block_5/multi_head_attention_5/key/kernel/momentum
P:N 2@SGD/transformer_block_5/multi_head_attention_5/key/bias/momentum
X:V  2DSGD/transformer_block_5/multi_head_attention_5/value/kernel/momentum
R:P 2BSGD/transformer_block_5/multi_head_attention_5/value/bias/momentum
c:a  2OSGD/transformer_block_5/multi_head_attention_5/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_5/multi_head_attention_5/attention_output/bias/momentum
,:* @2SGD/dense_16/kernel/momentum
&:$@2SGD/dense_16/bias/momentum
,:*@ 2SGD/dense_17/kernel/momentum
&:$ 2SGD/dense_17/bias/momentum
I:G 2=SGD/transformer_block_5/layer_normalization_10/gamma/momentum
H:F 2<SGD/transformer_block_5/layer_normalization_10/beta/momentum
I:G 2=SGD/transformer_block_5/layer_normalization_11/gamma/momentum
H:F 2<SGD/transformer_block_5/layer_normalization_11/beta/momentum
�2�
C__inference_model_2_layer_call_and_return_conditional_losses_187016
C__inference_model_2_layer_call_and_return_conditional_losses_186773
C__inference_model_2_layer_call_and_return_conditional_losses_185938
C__inference_model_2_layer_call_and_return_conditional_losses_186032�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_model_2_layer_call_fn_186377
(__inference_model_2_layer_call_fn_187172
(__inference_model_2_layer_call_fn_186205
(__inference_model_2_layer_call_fn_187094�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_184582�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *O�L
J�G
"�
input_5����������R
!�
input_6���������
�2�
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_187196�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
?__inference_token_and_position_embedding_2_layer_call_fn_187205�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv1d_4_layer_call_and_return_conditional_losses_187221�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv1d_4_layer_call_fn_187230�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_184591�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
4__inference_average_pooling1d_6_layer_call_fn_184597�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
D__inference_conv1d_5_layer_call_and_return_conditional_losses_187246�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv1d_5_layer_call_fn_187255�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_184606�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
4__inference_average_pooling1d_7_layer_call_fn_184612�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_184621�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
4__inference_average_pooling1d_8_layer_call_fn_184627�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+'���������������������������
�2�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187291
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187393
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187311
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187373�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_4_layer_call_fn_187406
6__inference_batch_normalization_4_layer_call_fn_187324
6__inference_batch_normalization_4_layer_call_fn_187337
6__inference_batch_normalization_4_layer_call_fn_187419�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187455
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187475
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187557
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187537�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
6__inference_batch_normalization_5_layer_call_fn_187488
6__inference_batch_normalization_5_layer_call_fn_187501
6__inference_batch_normalization_5_layer_call_fn_187570
6__inference_batch_normalization_5_layer_call_fn_187583�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_add_2_layer_call_and_return_conditional_losses_187589�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_add_2_layer_call_fn_187595�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_187743
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_187870�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
4__inference_transformer_block_5_layer_call_fn_187944
4__inference_transformer_block_5_layer_call_fn_187907�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults� 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_flatten_2_layer_call_and_return_conditional_losses_187950�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_flatten_2_layer_call_fn_187955�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_concatenate_2_layer_call_and_return_conditional_losses_187962�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_concatenate_2_layer_call_fn_187968�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_18_layer_call_and_return_conditional_losses_187979�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_18_layer_call_fn_187988�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dropout_16_layer_call_and_return_conditional_losses_188005
F__inference_dropout_16_layer_call_and_return_conditional_losses_188000�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_16_layer_call_fn_188015
+__inference_dropout_16_layer_call_fn_188010�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dense_19_layer_call_and_return_conditional_losses_188026�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_19_layer_call_fn_188035�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dropout_17_layer_call_and_return_conditional_losses_188047
F__inference_dropout_17_layer_call_and_return_conditional_losses_188052�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_dropout_17_layer_call_fn_188057
+__inference_dropout_17_layer_call_fn_188062�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_dense_20_layer_call_and_return_conditional_losses_188072�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_20_layer_call_fn_188081�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_186463input_5input_6"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_5_layer_call_and_return_conditional_losses_185005
H__inference_sequential_5_layer_call_and_return_conditional_losses_188195
H__inference_sequential_5_layer_call_and_return_conditional_losses_188138
H__inference_sequential_5_layer_call_and_return_conditional_losses_185019�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_sequential_5_layer_call_fn_188208
-__inference_sequential_5_layer_call_fn_185074
-__inference_sequential_5_layer_call_fn_188221
-__inference_sequential_5_layer_call_fn_185047�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_16_layer_call_and_return_conditional_losses_188252�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_16_layer_call_fn_188261�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_17_layer_call_and_return_conditional_losses_188291�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_17_layer_call_fn_188300�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_184582�4~ !*+<9;:EBDC����������������`ajktuY�V
O�L
J�G
"�
input_5����������R
!�
input_6���������
� "3�0
.
dense_20"�
dense_20����������
A__inference_add_2_layer_call_and_return_conditional_losses_187589�b�_
X�U
S�P
&�#
inputs/0���������# 
&�#
inputs/1���������# 
� ")�&
�
0���������# 
� �
&__inference_add_2_layer_call_fn_187595�b�_
X�U
S�P
&�#
inputs/0���������# 
&�#
inputs/1���������# 
� "����������# �
O__inference_average_pooling1d_6_layer_call_and_return_conditional_losses_184591�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
4__inference_average_pooling1d_6_layer_call_fn_184597wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
O__inference_average_pooling1d_7_layer_call_and_return_conditional_losses_184606�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
4__inference_average_pooling1d_7_layer_call_fn_184612wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
O__inference_average_pooling1d_8_layer_call_and_return_conditional_losses_184621�E�B
;�8
6�3
inputs'���������������������������
� ";�8
1�.
0'���������������������������
� �
4__inference_average_pooling1d_8_layer_call_fn_184627wE�B
;�8
6�3
inputs'���������������������������
� ".�+'����������������������������
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187291|;<9:@�=
6�3
-�*
inputs������������������ 
p
� "2�/
(�%
0������������������ 
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187311|<9;:@�=
6�3
-�*
inputs������������������ 
p 
� "2�/
(�%
0������������������ 
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187373j;<9:7�4
-�*
$�!
inputs���������# 
p
� ")�&
�
0���������# 
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_187393j<9;:7�4
-�*
$�!
inputs���������# 
p 
� ")�&
�
0���������# 
� �
6__inference_batch_normalization_4_layer_call_fn_187324o;<9:@�=
6�3
-�*
inputs������������������ 
p
� "%�"������������������ �
6__inference_batch_normalization_4_layer_call_fn_187337o<9;:@�=
6�3
-�*
inputs������������������ 
p 
� "%�"������������������ �
6__inference_batch_normalization_4_layer_call_fn_187406];<9:7�4
-�*
$�!
inputs���������# 
p
� "����������# �
6__inference_batch_normalization_4_layer_call_fn_187419]<9;:7�4
-�*
$�!
inputs���������# 
p 
� "����������# �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187455|DEBC@�=
6�3
-�*
inputs������������������ 
p
� "2�/
(�%
0������������������ 
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187475|EBDC@�=
6�3
-�*
inputs������������������ 
p 
� "2�/
(�%
0������������������ 
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187537jDEBC7�4
-�*
$�!
inputs���������# 
p
� ")�&
�
0���������# 
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_187557jEBDC7�4
-�*
$�!
inputs���������# 
p 
� ")�&
�
0���������# 
� �
6__inference_batch_normalization_5_layer_call_fn_187488oDEBC@�=
6�3
-�*
inputs������������������ 
p
� "%�"������������������ �
6__inference_batch_normalization_5_layer_call_fn_187501oEBDC@�=
6�3
-�*
inputs������������������ 
p 
� "%�"������������������ �
6__inference_batch_normalization_5_layer_call_fn_187570]DEBC7�4
-�*
$�!
inputs���������# 
p
� "����������# �
6__inference_batch_normalization_5_layer_call_fn_187583]EBDC7�4
-�*
$�!
inputs���������# 
p 
� "����������# �
I__inference_concatenate_2_layer_call_and_return_conditional_losses_187962�[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������
� "&�#
�
0����������
� �
.__inference_concatenate_2_layer_call_fn_187968x[�X
Q�N
L�I
#� 
inputs/0����������
"�
inputs/1���������
� "������������
D__inference_conv1d_4_layer_call_and_return_conditional_losses_187221g !4�1
*�'
%�"
inputs����������R 
� "+�(
!�
0����������R�
� �
)__inference_conv1d_4_layer_call_fn_187230Z !4�1
*�'
%�"
inputs����������R 
� "�����������R��
D__inference_conv1d_5_layer_call_and_return_conditional_losses_187246g*+5�2
+�(
&�#
inputs�����������
� "*�'
 �
0���������� 
� �
)__inference_conv1d_5_layer_call_fn_187255Z*+5�2
+�(
&�#
inputs�����������
� "����������� �
D__inference_dense_16_layer_call_and_return_conditional_losses_188252f��3�0
)�&
$�!
inputs���������# 
� ")�&
�
0���������#@
� �
)__inference_dense_16_layer_call_fn_188261Y��3�0
)�&
$�!
inputs���������# 
� "����������#@�
D__inference_dense_17_layer_call_and_return_conditional_losses_188291f��3�0
)�&
$�!
inputs���������#@
� ")�&
�
0���������# 
� �
)__inference_dense_17_layer_call_fn_188300Y��3�0
)�&
$�!
inputs���������#@
� "����������# �
D__inference_dense_18_layer_call_and_return_conditional_losses_187979]`a0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� }
)__inference_dense_18_layer_call_fn_187988P`a0�-
&�#
!�
inputs����������
� "����������@�
D__inference_dense_19_layer_call_and_return_conditional_losses_188026\jk/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_dense_19_layer_call_fn_188035Ojk/�,
%�"
 �
inputs���������@
� "����������@�
D__inference_dense_20_layer_call_and_return_conditional_losses_188072\tu/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� |
)__inference_dense_20_layer_call_fn_188081Otu/�,
%�"
 �
inputs���������@
� "�����������
F__inference_dropout_16_layer_call_and_return_conditional_losses_188000\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
F__inference_dropout_16_layer_call_and_return_conditional_losses_188005\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� ~
+__inference_dropout_16_layer_call_fn_188010O3�0
)�&
 �
inputs���������@
p
� "����������@~
+__inference_dropout_16_layer_call_fn_188015O3�0
)�&
 �
inputs���������@
p 
� "����������@�
F__inference_dropout_17_layer_call_and_return_conditional_losses_188047\3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
F__inference_dropout_17_layer_call_and_return_conditional_losses_188052\3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� ~
+__inference_dropout_17_layer_call_fn_188057O3�0
)�&
 �
inputs���������@
p
� "����������@~
+__inference_dropout_17_layer_call_fn_188062O3�0
)�&
 �
inputs���������@
p 
� "����������@�
E__inference_flatten_2_layer_call_and_return_conditional_losses_187950]3�0
)�&
$�!
inputs���������# 
� "&�#
�
0����������
� ~
*__inference_flatten_2_layer_call_fn_187955P3�0
)�&
$�!
inputs���������# 
� "������������
C__inference_model_2_layer_call_and_return_conditional_losses_185938�4~ !*+;<9:DEBC����������������`ajktua�^
W�T
J�G
"�
input_5����������R
!�
input_6���������
p

 
� "%�"
�
0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_186032�4~ !*+<9;:EBDC����������������`ajktua�^
W�T
J�G
"�
input_5����������R
!�
input_6���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_186773�4~ !*+;<9:DEBC����������������`ajktuc�`
Y�V
L�I
#� 
inputs/0����������R
"�
inputs/1���������
p

 
� "%�"
�
0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_187016�4~ !*+<9;:EBDC����������������`ajktuc�`
Y�V
L�I
#� 
inputs/0����������R
"�
inputs/1���������
p 

 
� "%�"
�
0���������
� �
(__inference_model_2_layer_call_fn_186205�4~ !*+;<9:DEBC����������������`ajktua�^
W�T
J�G
"�
input_5����������R
!�
input_6���������
p

 
� "�����������
(__inference_model_2_layer_call_fn_186377�4~ !*+<9;:EBDC����������������`ajktua�^
W�T
J�G
"�
input_5����������R
!�
input_6���������
p 

 
� "�����������
(__inference_model_2_layer_call_fn_187094�4~ !*+;<9:DEBC����������������`ajktuc�`
Y�V
L�I
#� 
inputs/0����������R
"�
inputs/1���������
p

 
� "�����������
(__inference_model_2_layer_call_fn_187172�4~ !*+<9;:EBDC����������������`ajktuc�`
Y�V
L�I
#� 
inputs/0����������R
"�
inputs/1���������
p 

 
� "�����������
H__inference_sequential_5_layer_call_and_return_conditional_losses_185005z����C�@
9�6
,�)
dense_16_input���������# 
p

 
� ")�&
�
0���������# 
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_185019z����C�@
9�6
,�)
dense_16_input���������# 
p 

 
� ")�&
�
0���������# 
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_188138r����;�8
1�.
$�!
inputs���������# 
p

 
� ")�&
�
0���������# 
� �
H__inference_sequential_5_layer_call_and_return_conditional_losses_188195r����;�8
1�.
$�!
inputs���������# 
p 

 
� ")�&
�
0���������# 
� �
-__inference_sequential_5_layer_call_fn_185047m����C�@
9�6
,�)
dense_16_input���������# 
p

 
� "����������# �
-__inference_sequential_5_layer_call_fn_185074m����C�@
9�6
,�)
dense_16_input���������# 
p 

 
� "����������# �
-__inference_sequential_5_layer_call_fn_188208e����;�8
1�.
$�!
inputs���������# 
p

 
� "����������# �
-__inference_sequential_5_layer_call_fn_188221e����;�8
1�.
$�!
inputs���������# 
p 

 
� "����������# �
$__inference_signature_wrapper_186463�4~ !*+<9;:EBDC����������������`ajktuj�g
� 
`�]
-
input_5"�
input_5����������R
,
input_6!�
input_6���������"3�0
.
dense_20"�
dense_20����������
Z__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_187196]~+�(
!�
�
x����������R
� "*�'
 �
0����������R 
� �
?__inference_token_and_position_embedding_2_layer_call_fn_187205P~+�(
!�
�
x����������R
� "�����������R �
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_187743� ����������������7�4
-�*
$�!
inputs���������# 
p
� ")�&
�
0���������# 
� �
O__inference_transformer_block_5_layer_call_and_return_conditional_losses_187870� ����������������7�4
-�*
$�!
inputs���������# 
p 
� ")�&
�
0���������# 
� �
4__inference_transformer_block_5_layer_call_fn_187907y ����������������7�4
-�*
$�!
inputs���������# 
p
� "����������# �
4__inference_transformer_block_5_layer_call_fn_187944y ����������������7�4
-�*
$�!
inputs���������# 
p 
� "����������# 