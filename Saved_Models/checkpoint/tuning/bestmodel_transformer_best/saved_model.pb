њЮ0
њ!Х!
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
Љ
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
Ы
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
≠
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
Н
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
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Н
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
dtypetypeИ
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
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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

2	Р
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Њ
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
executor_typestring И
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
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8сь)
~
conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv1d_6/kernel
w
#conv1d_6/kernel/Read/ReadVariableOpReadVariableOpconv1d_6/kernel*"
_output_shapes
:  *
dtype0
r
conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_6/bias
k
!conv1d_6/bias/Read/ReadVariableOpReadVariableOpconv1d_6/bias*
_output_shapes
: *
dtype0
~
conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  * 
shared_nameconv1d_7/kernel
w
#conv1d_7/kernel/Read/ReadVariableOpReadVariableOpconv1d_7/kernel*"
_output_shapes
:	  *
dtype0
r
conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_7/bias
k
!conv1d_7/bias/Read/ReadVariableOpReadVariableOpconv1d_7/bias*
_output_shapes
: *
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
{
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»@* 
shared_namedense_25/kernel
t
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes
:	»@*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:@*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:@@*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:@*
dtype0
z
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:@*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
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
∆
5token_and_position_embedding_3/embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_3/embedding_6/embeddings
њ
Itoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_6/embeddings*
_output_shapes

: *
dtype0
»
5token_and_position_embedding_3/embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†Ь *F
shared_name75token_and_position_embedding_3/embedding_7/embeddings
Ѕ
Itoken_and_position_embedding_3/embedding_7/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_3/embedding_7/embeddings* 
_output_shapes
:
†Ь *
dtype0
ќ
7transformer_block_7/multi_head_attention_7/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_7/multi_head_attention_7/query/kernel
«
Ktransformer_block_7/multi_head_attention_7/query/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_7/multi_head_attention_7/query/kernel*"
_output_shapes
:  *
dtype0
∆
5transformer_block_7/multi_head_attention_7/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_7/multi_head_attention_7/query/bias
њ
Itransformer_block_7/multi_head_attention_7/query/bias/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/query/bias*
_output_shapes

: *
dtype0
 
5transformer_block_7/multi_head_attention_7/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *F
shared_name75transformer_block_7/multi_head_attention_7/key/kernel
√
Itransformer_block_7/multi_head_attention_7/key/kernel/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/key/kernel*"
_output_shapes
:  *
dtype0
¬
3transformer_block_7/multi_head_attention_7/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *D
shared_name53transformer_block_7/multi_head_attention_7/key/bias
ї
Gtransformer_block_7/multi_head_attention_7/key/bias/Read/ReadVariableOpReadVariableOp3transformer_block_7/multi_head_attention_7/key/bias*
_output_shapes

: *
dtype0
ќ
7transformer_block_7/multi_head_attention_7/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *H
shared_name97transformer_block_7/multi_head_attention_7/value/kernel
«
Ktransformer_block_7/multi_head_attention_7/value/kernel/Read/ReadVariableOpReadVariableOp7transformer_block_7/multi_head_attention_7/value/kernel*"
_output_shapes
:  *
dtype0
∆
5transformer_block_7/multi_head_attention_7/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75transformer_block_7/multi_head_attention_7/value/bias
њ
Itransformer_block_7/multi_head_attention_7/value/bias/Read/ReadVariableOpReadVariableOp5transformer_block_7/multi_head_attention_7/value/bias*
_output_shapes

: *
dtype0
д
Btransformer_block_7/multi_head_attention_7/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBtransformer_block_7/multi_head_attention_7/attention_output/kernel
Ё
Vtransformer_block_7/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_7/multi_head_attention_7/attention_output/kernel*"
_output_shapes
:  *
dtype0
Ў
@transformer_block_7/multi_head_attention_7/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_7/multi_head_attention_7/attention_output/bias
—
Ttransformer_block_7/multi_head_attention_7/attention_output/bias/Read/ReadVariableOpReadVariableOp@transformer_block_7/multi_head_attention_7/attention_output/bias*
_output_shapes
: *
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

: @*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ * 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:@ *
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
: *
dtype0
Є
0transformer_block_7/layer_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_7/layer_normalization_14/gamma
±
Dtransformer_block_7/layer_normalization_14/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_7/layer_normalization_14/gamma*
_output_shapes
: *
dtype0
ґ
/transformer_block_7/layer_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_7/layer_normalization_14/beta
ѓ
Ctransformer_block_7/layer_normalization_14/beta/Read/ReadVariableOpReadVariableOp/transformer_block_7/layer_normalization_14/beta*
_output_shapes
: *
dtype0
Є
0transformer_block_7/layer_normalization_15/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_7/layer_normalization_15/gamma
±
Dtransformer_block_7/layer_normalization_15/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_7/layer_normalization_15/gamma*
_output_shapes
: *
dtype0
ґ
/transformer_block_7/layer_normalization_15/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_7/layer_normalization_15/beta
ѓ
Ctransformer_block_7/layer_normalization_15/beta/Read/ReadVariableOpReadVariableOp/transformer_block_7/layer_normalization_15/beta*
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
Ш
SGD/conv1d_6/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_nameSGD/conv1d_6/kernel/momentum
С
0SGD/conv1d_6/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_6/kernel/momentum*"
_output_shapes
:  *
dtype0
М
SGD/conv1d_6/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_6/bias/momentum
Е
.SGD/conv1d_6/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_6/bias/momentum*
_output_shapes
: *
dtype0
Ш
SGD/conv1d_7/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *-
shared_nameSGD/conv1d_7/kernel/momentum
С
0SGD/conv1d_7/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_7/kernel/momentum*"
_output_shapes
:	  *
dtype0
М
SGD/conv1d_7/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/conv1d_7/bias/momentum
Е
.SGD/conv1d_7/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/conv1d_7/bias/momentum*
_output_shapes
: *
dtype0
®
(SGD/batch_normalization_6/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_6/gamma/momentum
°
<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_6/gamma/momentum*
_output_shapes
: *
dtype0
¶
'SGD/batch_normalization_6/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_6/beta/momentum
Я
;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_6/beta/momentum*
_output_shapes
: *
dtype0
®
(SGD/batch_normalization_7/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(SGD/batch_normalization_7/gamma/momentum
°
<SGD/batch_normalization_7/gamma/momentum/Read/ReadVariableOpReadVariableOp(SGD/batch_normalization_7/gamma/momentum*
_output_shapes
: *
dtype0
¶
'SGD/batch_normalization_7/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'SGD/batch_normalization_7/beta/momentum
Я
;SGD/batch_normalization_7/beta/momentum/Read/ReadVariableOpReadVariableOp'SGD/batch_normalization_7/beta/momentum*
_output_shapes
: *
dtype0
Х
SGD/dense_25/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	»@*-
shared_nameSGD/dense_25/kernel/momentum
О
0SGD/dense_25/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_25/kernel/momentum*
_output_shapes
:	»@*
dtype0
М
SGD/dense_25/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_25/bias/momentum
Е
.SGD/dense_25/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_25/bias/momentum*
_output_shapes
:@*
dtype0
Ф
SGD/dense_26/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*-
shared_nameSGD/dense_26/kernel/momentum
Н
0SGD/dense_26/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_26/kernel/momentum*
_output_shapes

:@@*
dtype0
М
SGD/dense_26/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_26/bias/momentum
Е
.SGD/dense_26/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_26/bias/momentum*
_output_shapes
:@*
dtype0
Ф
SGD/dense_27/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_nameSGD/dense_27/kernel/momentum
Н
0SGD/dense_27/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_27/kernel/momentum*
_output_shapes

:@*
dtype0
М
SGD/dense_27/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameSGD/dense_27/bias/momentum
Е
.SGD/dense_27/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_27/bias/momentum*
_output_shapes
:*
dtype0
а
BSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum
ў
VSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum*
_output_shapes

: *
dtype0
в
BSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
†Ь *S
shared_nameDBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum
џ
VSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum/Read/ReadVariableOpReadVariableOpBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum* 
_output_shapes
:
†Ь *
dtype0
и
DSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum
б
XSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum*"
_output_shapes
:  *
dtype0
а
BSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum
ў
VSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum*
_output_shapes

: *
dtype0
д
BSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum
Ё
VSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum*"
_output_shapes
:  *
dtype0
№
@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *Q
shared_nameB@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum
’
TSGD/transformer_block_7/multi_head_attention_7/key/bias/momentum/Read/ReadVariableOpReadVariableOp@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum*
_output_shapes

: *
dtype0
и
DSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *U
shared_nameFDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum
б
XSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum/Read/ReadVariableOpReadVariableOpDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum*"
_output_shapes
:  *
dtype0
а
BSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum
ў
VSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum/Read/ReadVariableOpReadVariableOpBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum*
_output_shapes

: *
dtype0
ю
OSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *`
shared_nameQOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum
ч
cSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum/Read/ReadVariableOpReadVariableOpOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum*"
_output_shapes
:  *
dtype0
т
MSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *^
shared_nameOMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum
л
aSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum/Read/ReadVariableOpReadVariableOpMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum*
_output_shapes
: *
dtype0
Ф
SGD/dense_23/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*-
shared_nameSGD/dense_23/kernel/momentum
Н
0SGD/dense_23/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_23/kernel/momentum*
_output_shapes

: @*
dtype0
М
SGD/dense_23/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameSGD/dense_23/bias/momentum
Е
.SGD/dense_23/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_23/bias/momentum*
_output_shapes
:@*
dtype0
Ф
SGD/dense_24/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *-
shared_nameSGD/dense_24/kernel/momentum
Н
0SGD/dense_24/kernel/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_24/kernel/momentum*
_output_shapes

:@ *
dtype0
М
SGD/dense_24/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameSGD/dense_24/bias/momentum
Е
.SGD/dense_24/bias/momentum/Read/ReadVariableOpReadVariableOpSGD/dense_24/bias/momentum*
_output_shapes
: *
dtype0
“
=SGD/transformer_block_7/layer_normalization_14/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_7/layer_normalization_14/gamma/momentum
Ћ
QSGD/transformer_block_7/layer_normalization_14/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_7/layer_normalization_14/gamma/momentum*
_output_shapes
: *
dtype0
–
<SGD/transformer_block_7/layer_normalization_14/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_7/layer_normalization_14/beta/momentum
…
PSGD/transformer_block_7/layer_normalization_14/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_7/layer_normalization_14/beta/momentum*
_output_shapes
: *
dtype0
“
=SGD/transformer_block_7/layer_normalization_15/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=SGD/transformer_block_7/layer_normalization_15/gamma/momentum
Ћ
QSGD/transformer_block_7/layer_normalization_15/gamma/momentum/Read/ReadVariableOpReadVariableOp=SGD/transformer_block_7/layer_normalization_15/gamma/momentum*
_output_shapes
: *
dtype0
–
<SGD/transformer_block_7/layer_normalization_15/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><SGD/transformer_block_7/layer_normalization_15/beta/momentum
…
PSGD/transformer_block_7/layer_normalization_15/beta/momentum/Read/ReadVariableOpReadVariableOp<SGD/transformer_block_7/layer_normalization_15/beta/momentum*
_output_shapes
: *
dtype0

NoOpNoOp
Іґ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*бµ
value÷µB“µ B µ
џ
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
R
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api
R
4trainable_variables
5regularization_losses
6	variables
7	keras_api
Ч
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api
Ч
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
R
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
†
Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
 
R
\trainable_variables
]regularization_losses
^	variables
_	keras_api
h

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
R
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
h

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
R
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
h

tkernel
ubias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
ж
	zdecay
{learning_rate
|momentum
}iter momentumС!momentumТ*momentumУ+momentumФ9momentumХ:momentumЦBmomentumЧCmomentumШ`momentumЩamomentumЪjmomentumЫkmomentumЬtmomentumЭumomentumЮ~momentumЯmomentum†Аmomentum°БmomentumҐВmomentum£Гmomentum§Дmomentum•Еmomentum¶ЖmomentumІЗmomentum®Иmomentum©Йmomentum™КmomentumЂЛmomentumђМmomentum≠НmomentumЃОmomentumѓПmomentum∞
Ж
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
А10
Б11
В12
Г13
Д14
Е15
Ж16
З17
И18
Й19
К20
Л21
М22
Н23
О24
П25
`26
a27
j28
k29
t30
u31
 
¶
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
А14
Б15
В16
Г17
Д18
Е19
Ж20
З21
И22
Й23
К24
Л25
М26
Н27
О28
П29
`30
a31
j32
k33
t34
u35
≤
trainable_variables
Рnon_trainable_variables
Сmetrics
Тlayers
regularization_losses
Уlayer_metrics
 Фlayer_regularization_losses
	variables
 
f
~
embeddings
Хtrainable_variables
Цregularization_losses
Ч	variables
Ш	keras_api
f

embeddings
Щtrainable_variables
Ъregularization_losses
Ы	variables
Ь	keras_api

~0
1
 

~0
1
≤
Эmetrics
Юnon_trainable_variables
Яlayers
trainable_variables
regularization_losses
†layer_metrics
 °layer_regularization_losses
	variables
[Y
VARIABLE_VALUEconv1d_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
≤
Ґmetrics
£non_trainable_variables
§layers
"trainable_variables
#regularization_losses
•layer_metrics
 ¶layer_regularization_losses
$	variables
 
 
 
≤
Іmetrics
®non_trainable_variables
©layers
&trainable_variables
'regularization_losses
™layer_metrics
 Ђlayer_regularization_losses
(	variables
[Y
VARIABLE_VALUEconv1d_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
≤
ђmetrics
≠non_trainable_variables
Ѓlayers
,trainable_variables
-regularization_losses
ѓlayer_metrics
 ∞layer_regularization_losses
.	variables
 
 
 
≤
±metrics
≤non_trainable_variables
≥layers
0trainable_variables
1regularization_losses
іlayer_metrics
 µlayer_regularization_losses
2	variables
 
 
 
≤
ґmetrics
Јnon_trainable_variables
Єlayers
4trainable_variables
5regularization_losses
єlayer_metrics
 Їlayer_regularization_losses
6	variables
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
;2
<3
≤
їmetrics
Љnon_trainable_variables
љlayers
=trainable_variables
>regularization_losses
Њlayer_metrics
 њlayer_regularization_losses
?	variables
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 

B0
C1
D2
E3
≤
јmetrics
Ѕnon_trainable_variables
¬layers
Ftrainable_variables
Gregularization_losses
√layer_metrics
 ƒlayer_regularization_losses
H	variables
 
 
 
≤
≈metrics
∆non_trainable_variables
«layers
Jtrainable_variables
Kregularization_losses
»layer_metrics
 …layer_regularization_losses
L	variables
≈
 _query_dense
Ћ
_key_dense
ћ_value_dense
Ќ_softmax
ќ_dropout_layer
ѕ_output_dense
–trainable_variables
—regularization_losses
“	variables
”	keras_api
®
‘layer_with_weights-0
‘layer-0
’layer_with_weights-1
’layer-1
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
x
	Џaxis

Мgamma
	Нbeta
џtrainable_variables
№regularization_losses
Ё	variables
ё	keras_api
x
	яaxis

Оgamma
	Пbeta
аtrainable_variables
бregularization_losses
в	variables
г	keras_api
V
дtrainable_variables
еregularization_losses
ж	variables
з	keras_api
V
иtrainable_variables
йregularization_losses
к	variables
л	keras_api
Ж
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7
И8
Й9
К10
Л11
М12
Н13
О14
П15
 
Ж
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7
И8
Й9
К10
Л11
М12
Н13
О14
П15
≤
мmetrics
нnon_trainable_variables
оlayers
Ttrainable_variables
Uregularization_losses
пlayer_metrics
 рlayer_regularization_losses
V	variables
 
 
 
≤
сmetrics
тnon_trainable_variables
уlayers
Xtrainable_variables
Yregularization_losses
фlayer_metrics
 хlayer_regularization_losses
Z	variables
 
 
 
≤
цmetrics
чnon_trainable_variables
шlayers
\trainable_variables
]regularization_losses
щlayer_metrics
 ъlayer_regularization_losses
^	variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
 

`0
a1
≤
ыmetrics
ьnon_trainable_variables
эlayers
btrainable_variables
cregularization_losses
юlayer_metrics
 €layer_regularization_losses
d	variables
 
 
 
≤
Аmetrics
Бnon_trainable_variables
Вlayers
ftrainable_variables
gregularization_losses
Гlayer_metrics
 Дlayer_regularization_losses
h	variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1
 

j0
k1
≤
Еmetrics
Жnon_trainable_variables
Зlayers
ltrainable_variables
mregularization_losses
Иlayer_metrics
 Йlayer_regularization_losses
n	variables
 
 
 
≤
Кmetrics
Лnon_trainable_variables
Мlayers
ptrainable_variables
qregularization_losses
Нlayer_metrics
 Оlayer_regularization_losses
r	variables
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

t0
u1
 

t0
u1
≤
Пmetrics
Рnon_trainable_variables
Сlayers
vtrainable_variables
wregularization_losses
Тlayer_metrics
 Уlayer_regularization_losses
x	variables
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_3/embedding_6/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE5token_and_position_embedding_3/embedding_7/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7transformer_block_7/multi_head_attention_7/query/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/query/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/key/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3transformer_block_7/multi_head_attention_7/key/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE7transformer_block_7/multi_head_attention_7/value/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE5transformer_block_7/multi_head_attention_7/value/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUEBtransformer_block_7/multi_head_attention_7/attention_output/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE@transformer_block_7/multi_head_attention_7/attention_output/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_23/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_23/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_24/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_24/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_7/layer_normalization_14/gamma1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_7/layer_normalization_14/beta1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_7/layer_normalization_15/gamma1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_7/layer_normalization_15/beta1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE

;0
<1
D2
E3

Ф0
О
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
 
 

~0
 

~0
µ
Хmetrics
Цnon_trainable_variables
Чlayers
Хtrainable_variables
Цregularization_losses
Шlayer_metrics
 Щlayer_regularization_losses
Ч	variables

0
 

0
µ
Ъmetrics
Ыnon_trainable_variables
Ьlayers
Щtrainable_variables
Ъregularization_losses
Эlayer_metrics
 Юlayer_regularization_losses
Ы	variables
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
°
Яpartial_output_shape
†full_output_shape
Аkernel
	Бbias
°trainable_variables
Ґregularization_losses
£	variables
§	keras_api
°
•partial_output_shape
¶full_output_shape
Вkernel
	Гbias
Іtrainable_variables
®regularization_losses
©	variables
™	keras_api
°
Ђpartial_output_shape
ђfull_output_shape
Дkernel
	Еbias
≠trainable_variables
Ѓregularization_losses
ѓ	variables
∞	keras_api
V
±trainable_variables
≤regularization_losses
≥	variables
і	keras_api
V
µtrainable_variables
ґregularization_losses
Ј	variables
Є	keras_api
°
єpartial_output_shape
Їfull_output_shape
Жkernel
	Зbias
їtrainable_variables
Љregularization_losses
љ	variables
Њ	keras_api
@
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7
 
@
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7
µ
њmetrics
јnon_trainable_variables
Ѕlayers
–trainable_variables
—regularization_losses
¬layer_metrics
 √layer_regularization_losses
“	variables
n
Иkernel
	Йbias
ƒtrainable_variables
≈regularization_losses
∆	variables
«	keras_api
n
Кkernel
	Лbias
»trainable_variables
…regularization_losses
 	variables
Ћ	keras_api
 
И0
Й1
К2
Л3
 
 
И0
Й1
К2
Л3
µ
÷trainable_variables
ћnon_trainable_variables
Ќmetrics
ќlayers
„regularization_losses
ѕlayer_metrics
 –layer_regularization_losses
Ў	variables
 

М0
Н1
 

М0
Н1
µ
—metrics
“non_trainable_variables
”layers
џtrainable_variables
№regularization_losses
‘layer_metrics
 ’layer_regularization_losses
Ё	variables
 

О0
П1
 

О0
П1
µ
÷metrics
„non_trainable_variables
Ўlayers
аtrainable_variables
бregularization_losses
ўlayer_metrics
 Џlayer_regularization_losses
в	variables
 
 
 
µ
џmetrics
№non_trainable_variables
Ёlayers
дtrainable_variables
еregularization_losses
ёlayer_metrics
 яlayer_regularization_losses
ж	variables
 
 
 
µ
аmetrics
бnon_trainable_variables
вlayers
иtrainable_variables
йregularization_losses
гlayer_metrics
 дlayer_regularization_losses
к	variables
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
 
 
8

еtotal

жcount
з	variables
и	keras_api
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
А0
Б1
 

А0
Б1
µ
йmetrics
кnon_trainable_variables
лlayers
°trainable_variables
Ґregularization_losses
мlayer_metrics
 нlayer_regularization_losses
£	variables
 
 

В0
Г1
 

В0
Г1
µ
оmetrics
пnon_trainable_variables
рlayers
Іtrainable_variables
®regularization_losses
сlayer_metrics
 тlayer_regularization_losses
©	variables
 
 

Д0
Е1
 

Д0
Е1
µ
уmetrics
фnon_trainable_variables
хlayers
≠trainable_variables
Ѓregularization_losses
цlayer_metrics
 чlayer_regularization_losses
ѓ	variables
 
 
 
µ
шmetrics
щnon_trainable_variables
ъlayers
±trainable_variables
≤regularization_losses
ыlayer_metrics
 ьlayer_regularization_losses
≥	variables
 
 
 
µ
эmetrics
юnon_trainable_variables
€layers
µtrainable_variables
ґregularization_losses
Аlayer_metrics
 Бlayer_regularization_losses
Ј	variables
 
 

Ж0
З1
 

Ж0
З1
µ
Вmetrics
Гnon_trainable_variables
Дlayers
їtrainable_variables
Љregularization_losses
Еlayer_metrics
 Жlayer_regularization_losses
љ	variables
 
 
0
 0
Ћ1
ћ2
Ќ3
ќ4
ѕ5
 
 

И0
Й1
 

И0
Й1
µ
Зmetrics
Иnon_trainable_variables
Йlayers
ƒtrainable_variables
≈regularization_losses
Кlayer_metrics
 Лlayer_regularization_losses
∆	variables

К0
Л1
 

К0
Л1
µ
Мmetrics
Нnon_trainable_variables
Оlayers
»trainable_variables
…regularization_losses
Пlayer_metrics
 Рlayer_regularization_losses
 	variables
 
 

‘0
’1
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

е0
ж1

з	variables
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
МЙ
VARIABLE_VALUESGD/conv1d_6/kernel/momentumYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUESGD/conv1d_6/bias/momentumWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUESGD/conv1d_7/kernel/momentumYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUESGD/conv1d_7/bias/momentumWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE(SGD/batch_normalization_6/gamma/momentumXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE'SGD/batch_normalization_6/beta/momentumWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE(SGD/batch_normalization_7/gamma/momentumXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE'SGD/batch_normalization_7/beta/momentumWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUESGD/dense_25/kernel/momentumYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUESGD/dense_25/bias/momentumWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUESGD/dense_26/kernel/momentumYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUESGD/dense_26/bias/momentumWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUESGD/dense_27/kernel/momentumYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUESGD/dense_27/bias/momentumWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ђ©
VARIABLE_VALUEBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ђ©
VARIABLE_VALUEBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ѓђ
VARIABLE_VALUEDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
≠™
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
≠™
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentumTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Ђ®
VARIABLE_VALUE@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ѓђ
VARIABLE_VALUEDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
≠™
VARIABLE_VALUEBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЇЈ
VARIABLE_VALUEOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
Єµ
VARIABLE_VALUEMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUESGD/dense_23/kernel/momentumTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUESGD/dense_23/bias/momentumTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUESGD/dense_24/kernel/momentumTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUESGD/dense_24/bias/momentumTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®•
VARIABLE_VALUE=SGD/transformer_block_7/layer_normalization_14/gamma/momentumTtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
І§
VARIABLE_VALUE<SGD/transformer_block_7/layer_normalization_14/beta/momentumTtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
®•
VARIABLE_VALUE=SGD/transformer_block_7/layer_normalization_15/gamma/momentumTtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
І§
VARIABLE_VALUE<SGD/transformer_block_7/layer_normalization_15/beta/momentumTtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_input_7Placeholder*)
_output_shapes
:€€€€€€€€€†Ь*
dtype0*
shape:€€€€€€€€€†Ь
z
serving_default_input_8Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Г
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7serving_default_input_85token_and_position_embedding_3/embedding_7/embeddings5token_and_position_embedding_3/embedding_6/embeddingsconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/beta%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/beta7transformer_block_7/multi_head_attention_7/query/kernel5transformer_block_7/multi_head_attention_7/query/bias5transformer_block_7/multi_head_attention_7/key/kernel3transformer_block_7/multi_head_attention_7/key/bias7transformer_block_7/multi_head_attention_7/value/kernel5transformer_block_7/multi_head_attention_7/value/biasBtransformer_block_7/multi_head_attention_7/attention_output/kernel@transformer_block_7/multi_head_attention_7/attention_output/bias0transformer_block_7/layer_normalization_14/gamma/transformer_block_7/layer_normalization_14/betadense_23/kerneldense_23/biasdense_24/kerneldense_24/bias0transformer_block_7/layer_normalization_15/gamma/transformer_block_7/layer_normalization_15/betadense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_411175
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ћ$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_6/kernel/Read/ReadVariableOp!conv1d_6/bias/Read/ReadVariableOp#conv1d_7/kernel/Read/ReadVariableOp!conv1d_7/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpItoken_and_position_embedding_3/embedding_6/embeddings/Read/ReadVariableOpItoken_and_position_embedding_3/embedding_7/embeddings/Read/ReadVariableOpKtransformer_block_7/multi_head_attention_7/query/kernel/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/query/bias/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/key/kernel/Read/ReadVariableOpGtransformer_block_7/multi_head_attention_7/key/bias/Read/ReadVariableOpKtransformer_block_7/multi_head_attention_7/value/kernel/Read/ReadVariableOpItransformer_block_7/multi_head_attention_7/value/bias/Read/ReadVariableOpVtransformer_block_7/multi_head_attention_7/attention_output/kernel/Read/ReadVariableOpTtransformer_block_7/multi_head_attention_7/attention_output/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpDtransformer_block_7/layer_normalization_14/gamma/Read/ReadVariableOpCtransformer_block_7/layer_normalization_14/beta/Read/ReadVariableOpDtransformer_block_7/layer_normalization_15/gamma/Read/ReadVariableOpCtransformer_block_7/layer_normalization_15/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0SGD/conv1d_6/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_6/bias/momentum/Read/ReadVariableOp0SGD/conv1d_7/kernel/momentum/Read/ReadVariableOp.SGD/conv1d_7/bias/momentum/Read/ReadVariableOp<SGD/batch_normalization_6/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_6/beta/momentum/Read/ReadVariableOp<SGD/batch_normalization_7/gamma/momentum/Read/ReadVariableOp;SGD/batch_normalization_7/beta/momentum/Read/ReadVariableOp0SGD/dense_25/kernel/momentum/Read/ReadVariableOp.SGD/dense_25/bias/momentum/Read/ReadVariableOp0SGD/dense_26/kernel/momentum/Read/ReadVariableOp.SGD/dense_26/bias/momentum/Read/ReadVariableOp0SGD/dense_27/kernel/momentum/Read/ReadVariableOp.SGD/dense_27/bias/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum/Read/ReadVariableOpVSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum/Read/ReadVariableOpXSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum/Read/ReadVariableOpVSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum/Read/ReadVariableOpTSGD/transformer_block_7/multi_head_attention_7/key/bias/momentum/Read/ReadVariableOpXSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum/Read/ReadVariableOpVSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum/Read/ReadVariableOpcSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum/Read/ReadVariableOpaSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum/Read/ReadVariableOp0SGD/dense_23/kernel/momentum/Read/ReadVariableOp.SGD/dense_23/bias/momentum/Read/ReadVariableOp0SGD/dense_24/kernel/momentum/Read/ReadVariableOp.SGD/dense_24/bias/momentum/Read/ReadVariableOpQSGD/transformer_block_7/layer_normalization_14/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_7/layer_normalization_14/beta/momentum/Read/ReadVariableOpQSGD/transformer_block_7/layer_normalization_15/gamma/momentum/Read/ReadVariableOpPSGD/transformer_block_7/layer_normalization_15/beta/momentum/Read/ReadVariableOpConst*W
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_413258
€
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_6/kernelconv1d_6/biasconv1d_7/kernelconv1d_7/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variancebatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_25/kerneldense_25/biasdense_26/kerneldense_26/biasdense_27/kerneldense_27/biasdecaylearning_ratemomentumSGD/iter5token_and_position_embedding_3/embedding_6/embeddings5token_and_position_embedding_3/embedding_7/embeddings7transformer_block_7/multi_head_attention_7/query/kernel5transformer_block_7/multi_head_attention_7/query/bias5transformer_block_7/multi_head_attention_7/key/kernel3transformer_block_7/multi_head_attention_7/key/bias7transformer_block_7/multi_head_attention_7/value/kernel5transformer_block_7/multi_head_attention_7/value/biasBtransformer_block_7/multi_head_attention_7/attention_output/kernel@transformer_block_7/multi_head_attention_7/attention_output/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/bias0transformer_block_7/layer_normalization_14/gamma/transformer_block_7/layer_normalization_14/beta0transformer_block_7/layer_normalization_15/gamma/transformer_block_7/layer_normalization_15/betatotalcountSGD/conv1d_6/kernel/momentumSGD/conv1d_6/bias/momentumSGD/conv1d_7/kernel/momentumSGD/conv1d_7/bias/momentum(SGD/batch_normalization_6/gamma/momentum'SGD/batch_normalization_6/beta/momentum(SGD/batch_normalization_7/gamma/momentum'SGD/batch_normalization_7/beta/momentumSGD/dense_25/kernel/momentumSGD/dense_25/bias/momentumSGD/dense_26/kernel/momentumSGD/dense_26/bias/momentumSGD/dense_27/kernel/momentumSGD/dense_27/bias/momentumBSGD/token_and_position_embedding_3/embedding_6/embeddings/momentumBSGD/token_and_position_embedding_3/embedding_7/embeddings/momentumDSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentumBSGD/transformer_block_7/multi_head_attention_7/query/bias/momentumBSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentumDSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentumBSGD/transformer_block_7/multi_head_attention_7/value/bias/momentumOSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentumMSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentumSGD/dense_23/kernel/momentumSGD/dense_23/bias/momentumSGD/dense_24/kernel/momentumSGD/dense_24/bias/momentum=SGD/transformer_block_7/layer_normalization_14/gamma/momentum<SGD/transformer_block_7/layer_normalization_14/beta/momentum=SGD/transformer_block_7/layer_normalization_15/gamma/momentum<SGD/transformer_block_7/layer_normalization_15/beta/momentum*V
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_413490њљ&
ш
l
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_409333

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDimsЉ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize	
ђ*
paddingVALID*
strides	
ђ2	
AvgPoolО
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
єё
в
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412582

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_14/batchnorm/ReadVariableOpҐ3layer_normalization_14/batchnorm/mul/ReadVariableOpҐ/layer_normalization_15/batchnorm/ReadVariableOpҐ3layer_normalization_15/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_7/attention_output/add/ReadVariableOpҐDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_7/key/add/ReadVariableOpҐ7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/query/add/ReadVariableOpҐ9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/value/add/ReadVariableOpҐ9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐ,sequential_7/dense_23/BiasAdd/ReadVariableOpҐ.sequential_7/dense_23/Tensordot/ReadVariableOpҐ,sequential_7/dense_24/BiasAdd/ReadVariableOpҐ.sequential_7/dense_24/Tensordot/ReadVariableOpэ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/Einsumџ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpх
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/query/addч
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/Einsum’
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpн
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2 
multi_head_attention_7/key/addэ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/Einsumџ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpх
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/value/addБ
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_7/Mul/y∆
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 2
multi_head_attention_7/Mulь
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/Einsumƒ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2(
&multi_head_attention_7/softmax/Softmax 
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€BB2)
'multi_head_attention_7/dropout/IdentityФ
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/EinsumЮ
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumш
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOpЭ
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2-
+multi_head_attention_7/attention_output/addЭ
dropout_20/IdentityIdentity/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/Identityo
addAddV2inputsdropout_20/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
addЄ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesв
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_14/moments/meanќ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_14/moments/StopGradientо
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_14/moments/SquaredDifferenceј
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indicesЫ
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_14/moments/varianceХ
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_14/batchnorm/add/yо
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_14/batchnorm/addє
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_14/batchnorm/Rsqrtг
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpт
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/mulј
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_1е
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_2„
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpо
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/subе
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/add_1Ў
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOpЦ
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axesЭ
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free®
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape†
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axisњ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2§
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1Ш
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstЎ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/ProdЬ
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1а
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1Ь
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axisЮ
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatд
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackц
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2+
)sequential_7/dense_23/Tensordot/transposeч
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_23/Tensordot/Reshapeц
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_7/dense_23/Tensordot/MatMulЬ
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2†
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axisЂ
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1и
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2!
sequential_7/dense_23/Tensordotќ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpя
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/BiasAddЮ
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/ReluЎ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOpЦ
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axesЭ
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¶
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape†
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axisњ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2§
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1Ш
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstЎ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/ProdЬ
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1а
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1Ь
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axisЮ
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatд
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackф
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2+
)sequential_7/dense_24/Tensordot/transposeч
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_24/Tensordot/Reshapeц
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_7/dense_24/Tensordot/MatMulЬ
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2†
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axisЂ
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1и
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
sequential_7/dense_24/Tensordotќ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpя
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
sequential_7/dense_24/BiasAddФ
dropout_21/IdentityIdentity&sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/IdentityЧ
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
add_1Є
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesд
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_15/moments/meanќ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_15/moments/StopGradientр
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_15/moments/SquaredDifferenceј
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indicesЫ
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_15/moments/varianceХ
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_15/batchnorm/add/yо
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_15/batchnorm/addє
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_15/batchnorm/Rsqrtг
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpт
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/mul¬
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_1е
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_2„
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpо
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/subе
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/add_1№
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€B ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Ц
И
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412187

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
и
И
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409953

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
…
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_412764

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_409575

inputs
assignmovingavg_409550
assignmovingavg_1_409556)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409550*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409550AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409550*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_409556*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_409556AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409556*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
В
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_412759

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *эJБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ч
~
)__inference_conv1d_6_layer_call_fn_411942

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:€€€€€€€€€†Ь ::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€†Ь 
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412085

inputs
assignmovingavg_412060
assignmovingavg_1_412066)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_412060*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_412060AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412060*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_412066*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_412066AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412066*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ъ
ч
D__inference_conv1d_7_layer_call_and_return_conditional_losses_411958

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€Ъ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Ъ 
 
_user_specified_nameinputs
¬
u
I__inference_concatenate_3_layer_call_and_return_conditional_losses_412674
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisВ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€»2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€ј:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€ј
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
с	
Ё
D__inference_dense_25_layer_call_and_return_conditional_losses_412691

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
…
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_410610

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
№
§
(__inference_model_3_layer_call_fn_411884
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
identityИҐStatefulPartitionedCall÷
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
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4110142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:€€€€€€€€€†Ь
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
нX
Ю
C__inference_model_3_layer_call_and_return_conditional_losses_410744
input_7
input_8)
%token_and_position_embedding_3_410654)
%token_and_position_embedding_3_410656
conv1d_6_410659
conv1d_6_410661
conv1d_7_410665
conv1d_7_410667 
batch_normalization_6_410672 
batch_normalization_6_410674 
batch_normalization_6_410676 
batch_normalization_6_410678 
batch_normalization_7_410681 
batch_normalization_7_410683 
batch_normalization_7_410685 
batch_normalization_7_410687
transformer_block_7_410691
transformer_block_7_410693
transformer_block_7_410695
transformer_block_7_410697
transformer_block_7_410699
transformer_block_7_410701
transformer_block_7_410703
transformer_block_7_410705
transformer_block_7_410707
transformer_block_7_410709
transformer_block_7_410711
transformer_block_7_410713
transformer_block_7_410715
transformer_block_7_410717
transformer_block_7_410719
transformer_block_7_410721
dense_25_410726
dense_25_410728
dense_26_410732
dense_26_410734
dense_27_410738
dense_27_410740
identityИҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv1d_6/StatefulPartitionedCallҐ conv1d_7/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ6token_and_position_embedding_3/StatefulPartitionedCallҐ+transformer_block_7/StatefulPartitionedCallМ
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_410654%token_and_position_embedding_3_410656*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCall÷
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_410659conv1d_6_410661*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
 conv1d_6/StatefulPartitionedCall†
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCall¬
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_410665conv1d_7_410667*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
 conv1d_7/StatefulPartitionedCallЄ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
$average_pooling1d_11/PartitionedCallҐ
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCall√
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_410672batch_normalization_6_410674batch_normalization_6_410676batch_normalization_6_410678*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099532/
-batch_normalization_6/StatefulPartitionedCall√
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410681batch_normalization_7_410683batch_normalization_7_410685batch_normalization_7_410687*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100442/
-batch_normalization_7/StatefulPartitionedCallї
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCallО
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410691transformer_block_7_410693transformer_block_7_410695transformer_block_7_410697transformer_block_7_410699transformer_block_7_410701transformer_block_7_410703transformer_block_7_410705transformer_block_7_410707transformer_block_7_410709transformer_block_7_410711transformer_block_7_410713transformer_block_7_410715transformer_block_7_410717transformer_block_7_410719transformer_block_7_410721*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4103702-
+transformer_block_7/StatefulPartitionedCallЙ
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
flatten_3/PartitionedCallН
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0input_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCallЈ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410726dense_25_410728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
 dense_25/StatefulPartitionedCallА
dropout_22/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105532
dropout_22/PartitionedCallі
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_410732dense_26_410734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
 dense_26/StatefulPartitionedCallА
dropout_23/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106102
dropout_23/PartitionedCallі
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_410738dense_27_410740*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
 dense_27/StatefulPartitionedCallу
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:R N
)
_output_shapes
:€€€€€€€€€†Ь
!
_user_specified_name	input_7:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_8
єё
в
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_410370

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_14/batchnorm/ReadVariableOpҐ3layer_normalization_14/batchnorm/mul/ReadVariableOpҐ/layer_normalization_15/batchnorm/ReadVariableOpҐ3layer_normalization_15/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_7/attention_output/add/ReadVariableOpҐDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_7/key/add/ReadVariableOpҐ7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/query/add/ReadVariableOpҐ9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/value/add/ReadVariableOpҐ9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐ,sequential_7/dense_23/BiasAdd/ReadVariableOpҐ.sequential_7/dense_23/Tensordot/ReadVariableOpҐ,sequential_7/dense_24/BiasAdd/ReadVariableOpҐ.sequential_7/dense_24/Tensordot/ReadVariableOpэ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/Einsumџ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpх
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/query/addч
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/Einsum’
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpн
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2 
multi_head_attention_7/key/addэ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/Einsumџ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpх
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/value/addБ
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_7/Mul/y∆
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 2
multi_head_attention_7/Mulь
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/Einsumƒ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2(
&multi_head_attention_7/softmax/Softmax 
'multi_head_attention_7/dropout/IdentityIdentity0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€BB2)
'multi_head_attention_7/dropout/IdentityФ
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/Identity:output:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/EinsumЮ
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumш
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOpЭ
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2-
+multi_head_attention_7/attention_output/addЭ
dropout_20/IdentityIdentity/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/Identityo
addAddV2inputsdropout_20/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
addЄ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesв
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_14/moments/meanќ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_14/moments/StopGradientо
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_14/moments/SquaredDifferenceј
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indicesЫ
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_14/moments/varianceХ
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_14/batchnorm/add/yо
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_14/batchnorm/addє
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_14/batchnorm/Rsqrtг
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpт
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/mulј
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_1е
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_2„
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpо
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/subе
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/add_1Ў
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOpЦ
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axesЭ
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free®
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape†
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axisњ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2§
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1Ш
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstЎ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/ProdЬ
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1а
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1Ь
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axisЮ
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatд
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackц
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2+
)sequential_7/dense_23/Tensordot/transposeч
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_23/Tensordot/Reshapeц
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_7/dense_23/Tensordot/MatMulЬ
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2†
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axisЂ
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1и
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2!
sequential_7/dense_23/Tensordotќ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpя
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/BiasAddЮ
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/ReluЎ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOpЦ
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axesЭ
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¶
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape†
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axisњ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2§
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1Ш
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstЎ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/ProdЬ
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1а
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1Ь
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axisЮ
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatд
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackф
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2+
)sequential_7/dense_24/Tensordot/transposeч
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_24/Tensordot/Reshapeц
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_7/dense_24/Tensordot/MatMulЬ
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2†
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axisЂ
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1и
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
sequential_7/dense_24/Tensordotќ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpя
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
sequential_7/dense_24/BiasAddФ
dropout_21/IdentityIdentity&sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/IdentityЧ
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
add_1Є
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesд
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_15/moments/meanќ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_15/moments/StopGradientр
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_15/moments/SquaredDifferenceј
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indicesЫ
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_15/moments/varianceХ
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_15/batchnorm/add/yо
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_15/batchnorm/addє
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_15/batchnorm/Rsqrtг
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpт
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/mul¬
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_1е
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_2„
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpо
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/subе
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/add_1№
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€B ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Ц
И
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_409608

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ґ
ч
D__inference_conv1d_6_layer_call_and_return_conditional_losses_409847

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Є
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь *
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpО
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
Relu™
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:€€€€€€€€€†Ь ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€†Ь 
 
_user_specified_nameinputs
ц[
и
C__inference_model_3_layer_call_and_return_conditional_losses_410650
input_7
input_8)
%token_and_position_embedding_3_409826)
%token_and_position_embedding_3_409828
conv1d_6_409858
conv1d_6_409860
conv1d_7_409891
conv1d_7_409893 
batch_normalization_6_409980 
batch_normalization_6_409982 
batch_normalization_6_409984 
batch_normalization_6_409986 
batch_normalization_7_410071 
batch_normalization_7_410073 
batch_normalization_7_410075 
batch_normalization_7_410077
transformer_block_7_410446
transformer_block_7_410448
transformer_block_7_410450
transformer_block_7_410452
transformer_block_7_410454
transformer_block_7_410456
transformer_block_7_410458
transformer_block_7_410460
transformer_block_7_410462
transformer_block_7_410464
transformer_block_7_410466
transformer_block_7_410468
transformer_block_7_410470
transformer_block_7_410472
transformer_block_7_410474
transformer_block_7_410476
dense_25_410531
dense_25_410533
dense_26_410588
dense_26_410590
dense_27_410644
dense_27_410646
identityИҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv1d_6/StatefulPartitionedCallҐ conv1d_7/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ"dropout_22/StatefulPartitionedCallҐ"dropout_23/StatefulPartitionedCallҐ6token_and_position_embedding_3/StatefulPartitionedCallҐ+transformer_block_7/StatefulPartitionedCallМ
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_7%token_and_position_embedding_3_409826%token_and_position_embedding_3_409828*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCall÷
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_409858conv1d_6_409860*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
 conv1d_6/StatefulPartitionedCall†
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCall¬
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_409891conv1d_7_409893*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
 conv1d_7/StatefulPartitionedCallЄ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
$average_pooling1d_11/PartitionedCallҐ
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCallЅ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_409980batch_normalization_6_409982batch_normalization_6_409984batch_normalization_6_409986*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099332/
-batch_normalization_6/StatefulPartitionedCallЅ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410071batch_normalization_7_410073batch_normalization_7_410075batch_normalization_7_410077*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100242/
-batch_normalization_7/StatefulPartitionedCallї
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCallО
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410446transformer_block_7_410448transformer_block_7_410450transformer_block_7_410452transformer_block_7_410454transformer_block_7_410456transformer_block_7_410458transformer_block_7_410460transformer_block_7_410462transformer_block_7_410464transformer_block_7_410466transformer_block_7_410468transformer_block_7_410470transformer_block_7_410472transformer_block_7_410474transformer_block_7_410476*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4102432-
+transformer_block_7/StatefulPartitionedCallЙ
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
flatten_3/PartitionedCallН
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0input_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCallЈ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410531dense_25_410533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
 dense_25/StatefulPartitionedCallШ
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105482$
"dropout_22/StatefulPartitionedCallЉ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_410588dense_26_410590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
 dense_26/StatefulPartitionedCallљ
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106052$
"dropout_23/StatefulPartitionedCallЉ
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_410644dense_27_410646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
 dense_27/StatefulPartitionedCallљ
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:R N
)
_output_shapes
:€€€€€€€€€†Ь
!
_user_specified_name	input_7:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_8
Ц
И
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412105

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
—
г
D__inference_dense_24_layer_call_and_return_conditional_losses_413003

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2	
BiasAddЬ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€B@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B@
 
_user_specified_nameinputs
± 
г
D__inference_dense_23_layer_call_and_return_conditional_losses_409654

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
ReluЮ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€B ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Щ
G
+__inference_dropout_23_layer_call_fn_412774

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Т	
Ё
D__inference_dense_27_layer_call_and_return_conditional_losses_412784

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
о	
Ё
D__inference_dense_26_layer_call_and_return_conditional_losses_412738

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ж
Q
5__inference_average_pooling1d_10_layer_call_fn_409324

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т	
Ё
D__inference_dense_27_layer_call_and_return_conditional_losses_410633

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
®
Z
.__inference_concatenate_3_layer_call_fn_412680
inputs_0
inputs_1
identityЎ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€ј:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€ј
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
у0
»
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409435

inputs
assignmovingavg_409410
assignmovingavg_1_409416)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409410*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409410AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409410*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_409416*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_409416AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409416*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Є
†
-__inference_sequential_7_layer_call_fn_412920

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097482
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
б
~
)__inference_dense_25_layer_call_fn_412700

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
–
®
-__inference_sequential_7_layer_call_fn_409759
dense_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCalldense_23_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097482
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€B 
(
_user_specified_namedense_23_input
± 
г
D__inference_dense_23_layer_call_and_return_conditional_losses_412964

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
ReluЮ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€B ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Ж
Q
5__inference_average_pooling1d_11_layer_call_fn_409339

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
¬…
Щ1
"__inference__traced_restore_413490
file_prefix$
 assignvariableop_conv1d_6_kernel$
 assignvariableop_1_conv1d_6_bias&
"assignvariableop_2_conv1d_7_kernel$
 assignvariableop_3_conv1d_7_bias2
.assignvariableop_4_batch_normalization_6_gamma1
-assignvariableop_5_batch_normalization_6_beta8
4assignvariableop_6_batch_normalization_6_moving_mean<
8assignvariableop_7_batch_normalization_6_moving_variance2
.assignvariableop_8_batch_normalization_7_gamma1
-assignvariableop_9_batch_normalization_7_beta9
5assignvariableop_10_batch_normalization_7_moving_mean=
9assignvariableop_11_batch_normalization_7_moving_variance'
#assignvariableop_12_dense_25_kernel%
!assignvariableop_13_dense_25_bias'
#assignvariableop_14_dense_26_kernel%
!assignvariableop_15_dense_26_bias'
#assignvariableop_16_dense_27_kernel%
!assignvariableop_17_dense_27_bias
assignvariableop_18_decay%
!assignvariableop_19_learning_rate 
assignvariableop_20_momentum 
assignvariableop_21_sgd_iterM
Iassignvariableop_22_token_and_position_embedding_3_embedding_6_embeddingsM
Iassignvariableop_23_token_and_position_embedding_3_embedding_7_embeddingsO
Kassignvariableop_24_transformer_block_7_multi_head_attention_7_query_kernelM
Iassignvariableop_25_transformer_block_7_multi_head_attention_7_query_biasM
Iassignvariableop_26_transformer_block_7_multi_head_attention_7_key_kernelK
Gassignvariableop_27_transformer_block_7_multi_head_attention_7_key_biasO
Kassignvariableop_28_transformer_block_7_multi_head_attention_7_value_kernelM
Iassignvariableop_29_transformer_block_7_multi_head_attention_7_value_biasZ
Vassignvariableop_30_transformer_block_7_multi_head_attention_7_attention_output_kernelX
Tassignvariableop_31_transformer_block_7_multi_head_attention_7_attention_output_bias'
#assignvariableop_32_dense_23_kernel%
!assignvariableop_33_dense_23_bias'
#assignvariableop_34_dense_24_kernel%
!assignvariableop_35_dense_24_biasH
Dassignvariableop_36_transformer_block_7_layer_normalization_14_gammaG
Cassignvariableop_37_transformer_block_7_layer_normalization_14_betaH
Dassignvariableop_38_transformer_block_7_layer_normalization_15_gammaG
Cassignvariableop_39_transformer_block_7_layer_normalization_15_beta
assignvariableop_40_total
assignvariableop_41_count4
0assignvariableop_42_sgd_conv1d_6_kernel_momentum2
.assignvariableop_43_sgd_conv1d_6_bias_momentum4
0assignvariableop_44_sgd_conv1d_7_kernel_momentum2
.assignvariableop_45_sgd_conv1d_7_bias_momentum@
<assignvariableop_46_sgd_batch_normalization_6_gamma_momentum?
;assignvariableop_47_sgd_batch_normalization_6_beta_momentum@
<assignvariableop_48_sgd_batch_normalization_7_gamma_momentum?
;assignvariableop_49_sgd_batch_normalization_7_beta_momentum4
0assignvariableop_50_sgd_dense_25_kernel_momentum2
.assignvariableop_51_sgd_dense_25_bias_momentum4
0assignvariableop_52_sgd_dense_26_kernel_momentum2
.assignvariableop_53_sgd_dense_26_bias_momentum4
0assignvariableop_54_sgd_dense_27_kernel_momentum2
.assignvariableop_55_sgd_dense_27_bias_momentumZ
Vassignvariableop_56_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentumZ
Vassignvariableop_57_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum\
Xassignvariableop_58_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentumZ
Vassignvariableop_59_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentumZ
Vassignvariableop_60_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentumX
Tassignvariableop_61_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum\
Xassignvariableop_62_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentumZ
Vassignvariableop_63_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentumg
cassignvariableop_64_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentume
aassignvariableop_65_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum4
0assignvariableop_66_sgd_dense_23_kernel_momentum2
.assignvariableop_67_sgd_dense_23_bias_momentum4
0assignvariableop_68_sgd_dense_24_kernel_momentum2
.assignvariableop_69_sgd_dense_24_bias_momentumU
Qassignvariableop_70_sgd_transformer_block_7_layer_normalization_14_gamma_momentumT
Passignvariableop_71_sgd_transformer_block_7_layer_normalization_14_beta_momentumU
Qassignvariableop_72_sgd_transformer_block_7_layer_normalization_15_gamma_momentumT
Passignvariableop_73_sgd_transformer_block_7_layer_normalization_15_beta_momentum
identity_75ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_8ҐAssignVariableOp_9—(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ё'
value”'B–'KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesІ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ђ
value°BЮKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices•
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¬
_output_shapesѓ
ђ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_conv1d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1•
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4≥
AssignVariableOp_4AssignVariableOp.assignvariableop_4_batch_normalization_6_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5≤
AssignVariableOp_5AssignVariableOp-assignvariableop_5_batch_normalization_6_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6є
AssignVariableOp_6AssignVariableOp4assignvariableop_6_batch_normalization_6_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7љ
AssignVariableOp_7AssignVariableOp8assignvariableop_7_batch_normalization_6_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≥
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_7_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9≤
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_7_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10љ
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_7_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ѕ
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_7_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ђ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_25_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_25_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_26_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_26_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ђ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_27_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17©
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_27_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOpassignvariableop_18_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20§
AssignVariableOp_20AssignVariableOpassignvariableop_20_momentumIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21§
AssignVariableOp_21AssignVariableOpassignvariableop_21_sgd_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22—
AssignVariableOp_22AssignVariableOpIassignvariableop_22_token_and_position_embedding_3_embedding_6_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23—
AssignVariableOp_23AssignVariableOpIassignvariableop_23_token_and_position_embedding_3_embedding_7_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24”
AssignVariableOp_24AssignVariableOpKassignvariableop_24_transformer_block_7_multi_head_attention_7_query_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25—
AssignVariableOp_25AssignVariableOpIassignvariableop_25_transformer_block_7_multi_head_attention_7_query_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26—
AssignVariableOp_26AssignVariableOpIassignvariableop_26_transformer_block_7_multi_head_attention_7_key_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ѕ
AssignVariableOp_27AssignVariableOpGassignvariableop_27_transformer_block_7_multi_head_attention_7_key_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28”
AssignVariableOp_28AssignVariableOpKassignvariableop_28_transformer_block_7_multi_head_attention_7_value_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29—
AssignVariableOp_29AssignVariableOpIassignvariableop_29_transformer_block_7_multi_head_attention_7_value_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ё
AssignVariableOp_30AssignVariableOpVassignvariableop_30_transformer_block_7_multi_head_attention_7_attention_output_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31№
AssignVariableOp_31AssignVariableOpTassignvariableop_31_transformer_block_7_multi_head_attention_7_attention_output_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ђ
AssignVariableOp_32AssignVariableOp#assignvariableop_32_dense_23_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33©
AssignVariableOp_33AssignVariableOp!assignvariableop_33_dense_23_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ђ
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_24_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35©
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_24_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ћ
AssignVariableOp_36AssignVariableOpDassignvariableop_36_transformer_block_7_layer_normalization_14_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ћ
AssignVariableOp_37AssignVariableOpCassignvariableop_37_transformer_block_7_layer_normalization_14_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ћ
AssignVariableOp_38AssignVariableOpDassignvariableop_38_transformer_block_7_layer_normalization_15_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ћ
AssignVariableOp_39AssignVariableOpCassignvariableop_39_transformer_block_7_layer_normalization_15_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Є
AssignVariableOp_42AssignVariableOp0assignvariableop_42_sgd_conv1d_6_kernel_momentumIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43ґ
AssignVariableOp_43AssignVariableOp.assignvariableop_43_sgd_conv1d_6_bias_momentumIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Є
AssignVariableOp_44AssignVariableOp0assignvariableop_44_sgd_conv1d_7_kernel_momentumIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45ґ
AssignVariableOp_45AssignVariableOp.assignvariableop_45_sgd_conv1d_7_bias_momentumIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46ƒ
AssignVariableOp_46AssignVariableOp<assignvariableop_46_sgd_batch_normalization_6_gamma_momentumIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47√
AssignVariableOp_47AssignVariableOp;assignvariableop_47_sgd_batch_normalization_6_beta_momentumIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ƒ
AssignVariableOp_48AssignVariableOp<assignvariableop_48_sgd_batch_normalization_7_gamma_momentumIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49√
AssignVariableOp_49AssignVariableOp;assignvariableop_49_sgd_batch_normalization_7_beta_momentumIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Є
AssignVariableOp_50AssignVariableOp0assignvariableop_50_sgd_dense_25_kernel_momentumIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ґ
AssignVariableOp_51AssignVariableOp.assignvariableop_51_sgd_dense_25_bias_momentumIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Є
AssignVariableOp_52AssignVariableOp0assignvariableop_52_sgd_dense_26_kernel_momentumIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ґ
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sgd_dense_26_bias_momentumIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Є
AssignVariableOp_54AssignVariableOp0assignvariableop_54_sgd_dense_27_kernel_momentumIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55ґ
AssignVariableOp_55AssignVariableOp.assignvariableop_55_sgd_dense_27_bias_momentumIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56ё
AssignVariableOp_56AssignVariableOpVassignvariableop_56_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentumIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ё
AssignVariableOp_57AssignVariableOpVassignvariableop_57_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentumIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58а
AssignVariableOp_58AssignVariableOpXassignvariableop_58_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentumIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59ё
AssignVariableOp_59AssignVariableOpVassignvariableop_59_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentumIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ё
AssignVariableOp_60AssignVariableOpVassignvariableop_60_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentumIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61№
AssignVariableOp_61AssignVariableOpTassignvariableop_61_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentumIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62а
AssignVariableOp_62AssignVariableOpXassignvariableop_62_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentumIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63ё
AssignVariableOp_63AssignVariableOpVassignvariableop_63_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentumIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64л
AssignVariableOp_64AssignVariableOpcassignvariableop_64_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentumIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65й
AssignVariableOp_65AssignVariableOpaassignvariableop_65_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentumIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Є
AssignVariableOp_66AssignVariableOp0assignvariableop_66_sgd_dense_23_kernel_momentumIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67ґ
AssignVariableOp_67AssignVariableOp.assignvariableop_67_sgd_dense_23_bias_momentumIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Є
AssignVariableOp_68AssignVariableOp0assignvariableop_68_sgd_dense_24_kernel_momentumIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69ґ
AssignVariableOp_69AssignVariableOp.assignvariableop_69_sgd_dense_24_bias_momentumIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70ў
AssignVariableOp_70AssignVariableOpQassignvariableop_70_sgd_transformer_block_7_layer_normalization_14_gamma_momentumIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ў
AssignVariableOp_71AssignVariableOpPassignvariableop_71_sgd_transformer_block_7_layer_normalization_14_beta_momentumIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72ў
AssignVariableOp_72AssignVariableOpQassignvariableop_72_sgd_transformer_block_7_layer_normalization_15_gamma_momentumIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Ў
AssignVariableOp_73AssignVariableOpPassignvariableop_73_sgd_transformer_block_7_layer_normalization_15_beta_momentumIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_739
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_74≠
Identity_75IdentityIdentity_74:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_75"#
identity_75Identity_75:output:0*њ
_input_shapes≠
™: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
Ґ
ч
D__inference_conv1d_6_layer_call_and_return_conditional_losses_411933

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimШ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d/ExpandDims_1Є
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь *
paddingSAME*
strides
2
conv1dФ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpО
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2	
BiasAdd^
ReluReluBiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
Relu™
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:€€€€€€€€€†Ь ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€†Ь 
 
_user_specified_nameinputs
Щ
G
+__inference_dropout_22_layer_call_fn_412727

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ф[
и
C__inference_model_3_layer_call_and_return_conditional_losses_410842

inputs
inputs_1)
%token_and_position_embedding_3_410752)
%token_and_position_embedding_3_410754
conv1d_6_410757
conv1d_6_410759
conv1d_7_410763
conv1d_7_410765 
batch_normalization_6_410770 
batch_normalization_6_410772 
batch_normalization_6_410774 
batch_normalization_6_410776 
batch_normalization_7_410779 
batch_normalization_7_410781 
batch_normalization_7_410783 
batch_normalization_7_410785
transformer_block_7_410789
transformer_block_7_410791
transformer_block_7_410793
transformer_block_7_410795
transformer_block_7_410797
transformer_block_7_410799
transformer_block_7_410801
transformer_block_7_410803
transformer_block_7_410805
transformer_block_7_410807
transformer_block_7_410809
transformer_block_7_410811
transformer_block_7_410813
transformer_block_7_410815
transformer_block_7_410817
transformer_block_7_410819
dense_25_410824
dense_25_410826
dense_26_410830
dense_26_410832
dense_27_410836
dense_27_410838
identityИҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv1d_6/StatefulPartitionedCallҐ conv1d_7/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ"dropout_22/StatefulPartitionedCallҐ"dropout_23/StatefulPartitionedCallҐ6token_and_position_embedding_3/StatefulPartitionedCallҐ+transformer_block_7/StatefulPartitionedCallЛ
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_410752%token_and_position_embedding_3_410754*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCall÷
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_410757conv1d_6_410759*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
 conv1d_6/StatefulPartitionedCall†
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCall¬
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_410763conv1d_7_410765*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
 conv1d_7/StatefulPartitionedCallЄ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
$average_pooling1d_11/PartitionedCallҐ
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCallЅ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_410770batch_normalization_6_410772batch_normalization_6_410774batch_normalization_6_410776*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099332/
-batch_normalization_6/StatefulPartitionedCallЅ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410779batch_normalization_7_410781batch_normalization_7_410783batch_normalization_7_410785*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100242/
-batch_normalization_7/StatefulPartitionedCallї
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCallО
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410789transformer_block_7_410791transformer_block_7_410793transformer_block_7_410795transformer_block_7_410797transformer_block_7_410799transformer_block_7_410801transformer_block_7_410803transformer_block_7_410805transformer_block_7_410807transformer_block_7_410809transformer_block_7_410811transformer_block_7_410813transformer_block_7_410815transformer_block_7_410817transformer_block_7_410819*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4102432-
+transformer_block_7/StatefulPartitionedCallЙ
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
flatten_3/PartitionedCallО
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCallЈ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410824dense_25_410826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
 dense_25/StatefulPartitionedCallШ
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105482$
"dropout_22/StatefulPartitionedCallЉ
 dense_26/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_26_410830dense_26_410832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
 dense_26/StatefulPartitionedCallљ
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106052$
"dropout_23/StatefulPartitionedCallЉ
 dense_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_27_410836dense_27_410838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
 dense_27/StatefulPartitionedCallљ
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:Q M
)
_output_shapes
:€€€€€€€€€†Ь
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
В
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_410548

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *эJБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
и
И
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412023

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
°
F
*__inference_flatten_3_layer_call_fn_412667

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€B :S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
ђ
R
&__inference_add_3_layer_call_fn_412307
inputs_0
inputs_1
identity”
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:€€€€€€€€€B :€€€€€€€€€B :U Q
+
_output_shapes
:€€€€€€€€€B 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€B 
"
_user_specified_name
inputs/1
µ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_412662

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€B :S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
 
©
6__inference_batch_normalization_6_layer_call_fn_412049

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099532
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
м
©
6__inference_batch_normalization_6_layer_call_fn_412118

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4094352
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ЫС
и(
!__inference__wrapped_model_409294
input_7
input_8N
Jmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_409063N
Jmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_409069@
<model_3_conv1d_6_conv1d_expanddims_1_readvariableop_resource4
0model_3_conv1d_6_biasadd_readvariableop_resource@
<model_3_conv1d_7_conv1d_expanddims_1_readvariableop_resource4
0model_3_conv1d_7_biasadd_readvariableop_resourceC
?model_3_batch_normalization_6_batchnorm_readvariableop_resourceG
Cmodel_3_batch_normalization_6_batchnorm_mul_readvariableop_resourceE
Amodel_3_batch_normalization_6_batchnorm_readvariableop_1_resourceE
Amodel_3_batch_normalization_6_batchnorm_readvariableop_2_resourceC
?model_3_batch_normalization_7_batchnorm_readvariableop_resourceG
Cmodel_3_batch_normalization_7_batchnorm_mul_readvariableop_resourceE
Amodel_3_batch_normalization_7_batchnorm_readvariableop_1_resourceE
Amodel_3_batch_normalization_7_batchnorm_readvariableop_2_resourceb
^model_3_transformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resourceX
Tmodel_3_transformer_block_7_multi_head_attention_7_query_add_readvariableop_resource`
\model_3_transformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resourceV
Rmodel_3_transformer_block_7_multi_head_attention_7_key_add_readvariableop_resourceb
^model_3_transformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resourceX
Tmodel_3_transformer_block_7_multi_head_attention_7_value_add_readvariableop_resourcem
imodel_3_transformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resourcec
_model_3_transformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource\
Xmodel_3_transformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resourceX
Tmodel_3_transformer_block_7_layer_normalization_14_batchnorm_readvariableop_resourceW
Smodel_3_transformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resourceU
Qmodel_3_transformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resourceW
Smodel_3_transformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resourceU
Qmodel_3_transformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource\
Xmodel_3_transformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resourceX
Tmodel_3_transformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource3
/model_3_dense_25_matmul_readvariableop_resource4
0model_3_dense_25_biasadd_readvariableop_resource3
/model_3_dense_26_matmul_readvariableop_resource4
0model_3_dense_26_biasadd_readvariableop_resource3
/model_3_dense_27_matmul_readvariableop_resource4
0model_3_dense_27_biasadd_readvariableop_resource
identityИҐ6model_3/batch_normalization_6/batchnorm/ReadVariableOpҐ8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1Ґ8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2Ґ:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpҐ6model_3/batch_normalization_7/batchnorm/ReadVariableOpҐ8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1Ґ8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2Ґ:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpҐ'model_3/conv1d_6/BiasAdd/ReadVariableOpҐ3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpҐ'model_3/conv1d_7/BiasAdd/ReadVariableOpҐ3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpҐ'model_3/dense_25/BiasAdd/ReadVariableOpҐ&model_3/dense_25/MatMul/ReadVariableOpҐ'model_3/dense_26/BiasAdd/ReadVariableOpҐ&model_3/dense_26/MatMul/ReadVariableOpҐ'model_3/dense_27/BiasAdd/ReadVariableOpҐ&model_3/dense_27/MatMul/ReadVariableOpҐCmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupҐCmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupҐKmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpҐOmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpҐKmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpҐOmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpҐVmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpҐ`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐImodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpҐSmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐKmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpҐUmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐKmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpҐUmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐHmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpҐJmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpҐHmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpҐJmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpУ
,model_3/token_and_position_embedding_3/ShapeShapeinput_7*
T0*
_output_shapes
:2.
,model_3/token_and_position_embedding_3/ShapeЋ
:model_3/token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2<
:model_3/token_and_position_embedding_3/strided_slice/stack∆
<model_3/token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_3/token_and_position_embedding_3/strided_slice/stack_1∆
<model_3/token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_3/token_and_position_embedding_3/strided_slice/stack_2ћ
4model_3/token_and_position_embedding_3/strided_sliceStridedSlice5model_3/token_and_position_embedding_3/Shape:output:0Cmodel_3/token_and_position_embedding_3/strided_slice/stack:output:0Emodel_3/token_and_position_embedding_3/strided_slice/stack_1:output:0Emodel_3/token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_3/token_and_position_embedding_3/strided_slice™
2model_3/token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_3/token_and_position_embedding_3/range/start™
2model_3/token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_3/token_and_position_embedding_3/range/delta√
,model_3/token_and_position_embedding_3/rangeRange;model_3/token_and_position_embedding_3/range/start:output:0=model_3/token_and_position_embedding_3/strided_slice:output:0;model_3/token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2.
,model_3/token_and_position_embedding_3/rangeт
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_7_embedding_lookup_4090635model_3/token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/409063*'
_output_shapes
:€€€€€€€€€ *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupµ
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_7/embedding_lookup/409063*'
_output_shapes
:€€€€€€€€€ 2N
Lmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identityµ
Nmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityUmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2P
Nmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1∆
7model_3/token_and_position_embedding_3/embedding_6/CastCastinput_7*

DstT0*

SrcT0*)
_output_shapes
:€€€€€€€€€†Ь29
7model_3/token_and_position_embedding_3/embedding_6/Castю
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherJmodel_3_token_and_position_embedding_3_embedding_6_embedding_lookup_409069;model_3/token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/409069*-
_output_shapes
:€€€€€€€€€†Ь *
dtype02E
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupї
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityLmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_3/token_and_position_embedding_3/embedding_6/embedding_lookup/409069*-
_output_shapes
:€€€€€€€€€†Ь 2N
Lmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identityї
Nmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityUmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2P
Nmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1Ћ
*model_3/token_and_position_embedding_3/addAddV2Wmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Wmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2,
*model_3/token_and_position_embedding_3/addЫ
&model_3/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2(
&model_3/conv1d_6/conv1d/ExpandDims/dimу
"model_3/conv1d_6/conv1d/ExpandDims
ExpandDims.model_3/token_and_position_embedding_3/add:z:0/model_3/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2$
"model_3/conv1d_6/conv1d/ExpandDimsл
3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_3_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype025
3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЦ
(model_3/conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_3/conv1d_6/conv1d/ExpandDims_1/dimы
$model_3/conv1d_6/conv1d/ExpandDims_1
ExpandDims;model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:01model_3/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2&
$model_3/conv1d_6/conv1d/ExpandDims_1ь
model_3/conv1d_6/conv1dConv2D+model_3/conv1d_6/conv1d/ExpandDims:output:0-model_3/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь *
paddingSAME*
strides
2
model_3/conv1d_6/conv1d«
model_3/conv1d_6/conv1d/SqueezeSqueeze model_3/conv1d_6/conv1d:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь *
squeeze_dims

э€€€€€€€€2!
model_3/conv1d_6/conv1d/Squeezeњ
'model_3/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp0model_3_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/conv1d_6/BiasAdd/ReadVariableOp“
model_3/conv1d_6/BiasAddBiasAdd(model_3/conv1d_6/conv1d/Squeeze:output:0/model_3/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
model_3/conv1d_6/BiasAddС
model_3/conv1d_6/ReluRelu!model_3/conv1d_6/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
model_3/conv1d_6/ReluЪ
*model_3/average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_3/average_pooling1d_9/ExpandDims/dimф
&model_3/average_pooling1d_9/ExpandDims
ExpandDims#model_3/conv1d_6/Relu:activations:03model_3/average_pooling1d_9/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2(
&model_3/average_pooling1d_9/ExpandDimsэ
#model_3/average_pooling1d_9/AvgPoolAvgPool/model_3/average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
ksize
*
paddingVALID*
strides
2%
#model_3/average_pooling1d_9/AvgPool—
#model_3/average_pooling1d_9/SqueezeSqueeze,model_3/average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims
2%
#model_3/average_pooling1d_9/SqueezeЫ
&model_3/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2(
&model_3/conv1d_7/conv1d/ExpandDims/dimр
"model_3/conv1d_7/conv1d/ExpandDims
ExpandDims,model_3/average_pooling1d_9/Squeeze:output:0/model_3/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2$
"model_3/conv1d_7/conv1d/ExpandDimsл
3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<model_3_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype025
3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЦ
(model_3/conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(model_3/conv1d_7/conv1d/ExpandDims_1/dimы
$model_3/conv1d_7/conv1d/ExpandDims_1
ExpandDims;model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:01model_3/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2&
$model_3/conv1d_7/conv1d/ExpandDims_1ы
model_3/conv1d_7/conv1dConv2D+model_3/conv1d_7/conv1d/ExpandDims:output:0-model_3/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
paddingSAME*
strides
2
model_3/conv1d_7/conv1d∆
model_3/conv1d_7/conv1d/SqueezeSqueeze model_3/conv1d_7/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims

э€€€€€€€€2!
model_3/conv1d_7/conv1d/Squeezeњ
'model_3/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp0model_3_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_3/conv1d_7/BiasAdd/ReadVariableOp—
model_3/conv1d_7/BiasAddBiasAdd(model_3/conv1d_7/conv1d/Squeeze:output:0/model_3/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
model_3/conv1d_7/BiasAddР
model_3/conv1d_7/ReluRelu!model_3/conv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
model_3/conv1d_7/ReluЬ
+model_3/average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_3/average_pooling1d_11/ExpandDims/dimВ
'model_3/average_pooling1d_11/ExpandDims
ExpandDims.model_3/token_and_position_embedding_3/add:z:04model_3/average_pooling1d_11/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2)
'model_3/average_pooling1d_11/ExpandDimsБ
$model_3/average_pooling1d_11/AvgPoolAvgPool0model_3/average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€B *
ksize	
ђ*
paddingVALID*
strides	
ђ2&
$model_3/average_pooling1d_11/AvgPool”
$model_3/average_pooling1d_11/SqueezeSqueeze-model_3/average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
squeeze_dims
2&
$model_3/average_pooling1d_11/SqueezeЬ
+model_3/average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_3/average_pooling1d_10/ExpandDims/dimц
'model_3/average_pooling1d_10/ExpandDims
ExpandDims#model_3/conv1d_7/Relu:activations:04model_3/average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2)
'model_3/average_pooling1d_10/ExpandDims€
$model_3/average_pooling1d_10/AvgPoolAvgPool0model_3/average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€B *
ksize

*
paddingVALID*
strides

2&
$model_3/average_pooling1d_10/AvgPool”
$model_3/average_pooling1d_10/SqueezeSqueeze-model_3/average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
squeeze_dims
2&
$model_3/average_pooling1d_10/Squeezeм
6model_3/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_3_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_3/batch_normalization_6/batchnorm/ReadVariableOp£
-model_3/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_3/batch_normalization_6/batchnorm/add/yА
+model_3/batch_normalization_6/batchnorm/addAddV2>model_3/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_3/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/addљ
-model_3/batch_normalization_6/batchnorm/RsqrtRsqrt/model_3/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_6/batchnorm/Rsqrtш
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOpэ
+model_3/batch_normalization_6/batchnorm/mulMul1model_3/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/mulы
-model_3/batch_normalization_6/batchnorm/mul_1Mul-model_3/average_pooling1d_10/Squeeze:output:0/model_3/batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2/
-model_3/batch_normalization_6/batchnorm/mul_1т
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_1э
-model_3/batch_normalization_6/batchnorm/mul_2Mul@model_3/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_6/batchnorm/mul_2т
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_2ы
+model_3/batch_normalization_6/batchnorm/subSub@model_3/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_6/batchnorm/subБ
-model_3/batch_normalization_6/batchnorm/add_1AddV21model_3/batch_normalization_6/batchnorm/mul_1:z:0/model_3/batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2/
-model_3/batch_normalization_6/batchnorm/add_1м
6model_3/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_3_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6model_3/batch_normalization_7/batchnorm/ReadVariableOp£
-model_3/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_3/batch_normalization_7/batchnorm/add/yА
+model_3/batch_normalization_7/batchnorm/addAddV2>model_3/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_3/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/addљ
-model_3/batch_normalization_7/batchnorm/RsqrtRsqrt/model_3/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_7/batchnorm/Rsqrtш
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_3_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOpэ
+model_3/batch_normalization_7/batchnorm/mulMul1model_3/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_3/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/mulы
-model_3/batch_normalization_7/batchnorm/mul_1Mul-model_3/average_pooling1d_11/Squeeze:output:0/model_3/batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2/
-model_3/batch_normalization_7/batchnorm/mul_1т
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_3_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_1э
-model_3/batch_normalization_7/batchnorm/mul_2Mul@model_3/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_3/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-model_3/batch_normalization_7/batchnorm/mul_2т
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_3_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_2ы
+model_3/batch_normalization_7/batchnorm/subSub@model_3/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_3/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+model_3/batch_normalization_7/batchnorm/subБ
-model_3/batch_normalization_7/batchnorm/add_1AddV21model_3/batch_normalization_7/batchnorm/mul_1:z:0/model_3/batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2/
-model_3/batch_normalization_7/batchnorm/add_1Ћ
model_3/add_3/addAddV21model_3/batch_normalization_6/batchnorm/add_1:z:01model_3/batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
model_3/add_3/add—
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOp^model_3_transformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpр
Fmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsummodel_3/add_3/add:z:0]model_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2H
Fmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsumѓ
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpе
<model_3/transformer_block_7/multi_head_attention_7/query/addAddV2Omodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Smodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2>
<model_3/transformer_block_7/multi_head_attention_7/query/addЋ
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp\model_3_transformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02U
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpк
Dmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsummodel_3/add_3/add:z:0[model_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2F
Dmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum©
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpRmodel_3_transformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02K
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpЁ
:model_3/transformer_block_7/multi_head_attention_7/key/addAddV2Mmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Qmodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2<
:model_3/transformer_block_7/multi_head_attention_7/key/add—
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOp^model_3_transformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02W
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpр
Fmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsummodel_3/add_3/add:z:0]model_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2H
Fmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsumѓ
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02M
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpе
<model_3/transformer_block_7/multi_head_attention_7/value/addAddV2Omodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Smodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2>
<model_3/transformer_block_7/multi_head_attention_7/value/addє
8model_3/transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2:
8model_3/transformer_block_7/multi_head_attention_7/Mul/yґ
6model_3/transformer_block_7/multi_head_attention_7/MulMul@model_3/transformer_block_7/multi_head_attention_7/query/add:z:0Amodel_3/transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 28
6model_3/transformer_block_7/multi_head_attention_7/Mulм
@model_3/transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum>model_3/transformer_block_7/multi_head_attention_7/key/add:z:0:model_3/transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2B
@model_3/transformer_block_7/multi_head_attention_7/einsum/EinsumШ
Bmodel_3/transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxImodel_3/transformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2D
Bmodel_3/transformer_block_7/multi_head_attention_7/softmax/SoftmaxЮ
Cmodel_3/transformer_block_7/multi_head_attention_7/dropout/IdentityIdentityLmodel_3/transformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€BB2E
Cmodel_3/transformer_block_7/multi_head_attention_7/dropout/IdentityД
Bmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumLmodel_3/transformer_block_7/multi_head_attention_7/dropout/Identity:output:0@model_3/transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2D
Bmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/Einsumт
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpimodel_3_transformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02b
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp√
Qmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumKmodel_3/transformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0hmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe2S
Qmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsumћ
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOp_model_3_transformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02X
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpН
Gmodel_3/transformer_block_7/multi_head_attention_7/attention_output/addAddV2Zmodel_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0^model_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2I
Gmodel_3/transformer_block_7/multi_head_attention_7/attention_output/addс
/model_3/transformer_block_7/dropout_20/IdentityIdentityKmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 21
/model_3/transformer_block_7/dropout_20/Identity“
model_3/transformer_block_7/addAddV2model_3/add_3/add:z:08model_3/transformer_block_7/dropout_20/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
model_3/transformer_block_7/addр
Qmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indices“
?model_3/transformer_block_7/layer_normalization_14/moments/meanMean#model_3/transformer_block_7/add:z:0Zmodel_3/transformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2A
?model_3/transformer_block_7/layer_normalization_14/moments/meanҐ
Gmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradientStopGradientHmodel_3/transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2I
Gmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradientё
Lmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifference#model_3/transformer_block_7/add:z:0Pmodel_3/transformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2N
Lmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifferenceш
Umodel_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indicesЛ
Cmodel_3/transformer_block_7/layer_normalization_14/moments/varianceMeanPmodel_3/transformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0^model_3/transformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2E
Cmodel_3/transformer_block_7/layer_normalization_14/moments/varianceЌ
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/yё
@model_3/transformer_block_7/layer_normalization_14/batchnorm/addAddV2Lmodel_3/transformer_block_7/layer_normalization_14/moments/variance:output:0Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/addН
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrtDmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/RsqrtЈ
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_3_transformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpв
@model_3/transformer_block_7/layer_normalization_14/batchnorm/mulMulFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Wmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/mul∞
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1Mul#model_3/transformer_block_7/add:z:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1’
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2MulHmodel_3/transformer_block_7/layer_normalization_14/moments/mean:output:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2Ђ
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpё
@model_3/transformer_block_7/layer_normalization_14/batchnorm/subSubSmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2B
@model_3/transformer_block_7/layer_normalization_14/batchnorm/sub’
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0Dmodel_3/transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2D
Bmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1ђ
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpSmodel_3_transformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02L
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpќ
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes’
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/freeь
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShapeFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ShapeЎ
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisЋ
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2№
Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis—
Fmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Tmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1–
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const»
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdMmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod‘
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1–
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdOmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Lmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1‘
Gmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axis™
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Pmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat‘
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stackPackImodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stackж
Emodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transpose	TransposeFmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Kmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2G
Emodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transposeз
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeImodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Reshapeж
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulLmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2D
Bmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMul‘
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2E
Cmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Ў
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisЈ
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1ConcatV2Mmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Lmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/Const_2:output:0Rmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Ў
;model_3/transformer_block_7/sequential_7/dense_23/TensordotReshapeLmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Mmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2=
;model_3/transformer_block_7/sequential_7/dense_23/TensordotҐ
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpQmodel_3_transformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpѕ
9model_3/transformer_block_7/sequential_7/dense_23/BiasAddBiasAddDmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot:output:0Pmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2;
9model_3/transformer_block_7/sequential_7/dense_23/BiasAddт
6model_3/transformer_block_7/sequential_7/dense_23/ReluReluBmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@28
6model_3/transformer_block_7/sequential_7/dense_23/Reluђ
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpSmodel_3_transformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02L
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpќ
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes’
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/freeъ
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShapeDmodel_3/transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ShapeЎ
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisЋ
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2№
Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis—
Fmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Tmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1–
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const»
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdMmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod‘
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1–
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdOmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Lmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1‘
Gmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axis™
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Pmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat‘
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stackPackImodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stackд
Emodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transpose	TransposeDmodel_3/transformer_block_7/sequential_7/dense_23/Relu:activations:0Kmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2G
Emodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transposeз
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeImodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Reshapeж
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulLmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2D
Bmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMul‘
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Ў
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisЈ
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1ConcatV2Mmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Lmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/Const_2:output:0Rmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Ў
;model_3/transformer_block_7/sequential_7/dense_24/TensordotReshapeLmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Mmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2=
;model_3/transformer_block_7/sequential_7/dense_24/TensordotҐ
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpQmodel_3_transformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpѕ
9model_3/transformer_block_7/sequential_7/dense_24/BiasAddBiasAddDmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot:output:0Pmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2;
9model_3/transformer_block_7/sequential_7/dense_24/BiasAddи
/model_3/transformer_block_7/dropout_21/IdentityIdentityBmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 21
/model_3/transformer_block_7/dropout_21/IdentityЗ
!model_3/transformer_block_7/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_14/batchnorm/add_1:z:08model_3/transformer_block_7/dropout_21/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2#
!model_3/transformer_block_7/add_1р
Qmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indices‘
?model_3/transformer_block_7/layer_normalization_15/moments/meanMean%model_3/transformer_block_7/add_1:z:0Zmodel_3/transformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2A
?model_3/transformer_block_7/layer_normalization_15/moments/meanҐ
Gmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradientStopGradientHmodel_3/transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2I
Gmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradientа
Lmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifference%model_3/transformer_block_7/add_1:z:0Pmodel_3/transformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2N
Lmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifferenceш
Umodel_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indicesЛ
Cmodel_3/transformer_block_7/layer_normalization_15/moments/varianceMeanPmodel_3/transformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0^model_3/transformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2E
Cmodel_3/transformer_block_7/layer_normalization_15/moments/varianceЌ
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/yё
@model_3/transformer_block_7/layer_normalization_15/batchnorm/addAddV2Lmodel_3/transformer_block_7/layer_normalization_15/moments/variance:output:0Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/addН
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrtDmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/RsqrtЈ
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_3_transformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpв
@model_3/transformer_block_7/layer_normalization_15/batchnorm/mulMulFmodel_3/transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Wmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/mul≤
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1Mul%model_3/transformer_block_7/add_1:z:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1’
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2MulHmodel_3/transformer_block_7/layer_normalization_15/moments/mean:output:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2Ђ
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpTmodel_3_transformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpё
@model_3/transformer_block_7/layer_normalization_15/batchnorm/subSubSmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0Fmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2B
@model_3/transformer_block_7/layer_normalization_15/batchnorm/sub’
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2Fmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0Dmodel_3/transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2D
Bmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1Г
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
model_3/flatten_3/Constё
model_3/flatten_3/ReshapeReshapeFmodel_3/transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0 model_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
model_3/flatten_3/ReshapeИ
!model_3/concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_3/concatenate_3/concat/axisЁ
model_3/concatenate_3/concatConcatV2"model_3/flatten_3/Reshape:output:0input_8*model_3/concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€»2
model_3/concatenate_3/concatЅ
&model_3/dense_25/MatMul/ReadVariableOpReadVariableOp/model_3_dense_25_matmul_readvariableop_resource*
_output_shapes
:	»@*
dtype02(
&model_3/dense_25/MatMul/ReadVariableOp≈
model_3/dense_25/MatMulMatMul%model_3/concatenate_3/concat:output:0.model_3/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dense_25/MatMulњ
'model_3/dense_25/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_25/BiasAdd/ReadVariableOp≈
model_3/dense_25/BiasAddBiasAdd!model_3/dense_25/MatMul:product:0/model_3/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dense_25/BiasAddЛ
model_3/dense_25/ReluRelu!model_3/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dense_25/ReluЭ
model_3/dropout_22/IdentityIdentity#model_3/dense_25/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dropout_22/Identityј
&model_3/dense_26/MatMul/ReadVariableOpReadVariableOp/model_3_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_3/dense_26/MatMul/ReadVariableOpƒ
model_3/dense_26/MatMulMatMul$model_3/dropout_22/Identity:output:0.model_3/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dense_26/MatMulњ
'model_3/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_3/dense_26/BiasAdd/ReadVariableOp≈
model_3/dense_26/BiasAddBiasAdd!model_3/dense_26/MatMul:product:0/model_3/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dense_26/BiasAddЛ
model_3/dense_26/ReluRelu!model_3/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dense_26/ReluЭ
model_3/dropout_23/IdentityIdentity#model_3/dense_26/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
model_3/dropout_23/Identityј
&model_3/dense_27/MatMul/ReadVariableOpReadVariableOp/model_3_dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_3/dense_27/MatMul/ReadVariableOpƒ
model_3/dense_27/MatMulMatMul$model_3/dropout_23/Identity:output:0.model_3/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_3/dense_27/MatMulњ
'model_3/dense_27/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_3/dense_27/BiasAdd/ReadVariableOp≈
model_3/dense_27/BiasAddBiasAdd!model_3/dense_27/MatMul:product:0/model_3/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model_3/dense_27/BiasAddђ
IdentityIdentity!model_3/dense_27/BiasAdd:output:07^model_3/batch_normalization_6/batchnorm/ReadVariableOp9^model_3/batch_normalization_6/batchnorm/ReadVariableOp_19^model_3/batch_normalization_6/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp7^model_3/batch_normalization_7/batchnorm/ReadVariableOp9^model_3/batch_normalization_7/batchnorm/ReadVariableOp_19^model_3/batch_normalization_7/batchnorm/ReadVariableOp_2;^model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp(^model_3/conv1d_6/BiasAdd/ReadVariableOp4^model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp(^model_3/conv1d_7/BiasAdd/ReadVariableOp4^model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp(^model_3/dense_25/BiasAdd/ReadVariableOp'^model_3/dense_25/MatMul/ReadVariableOp(^model_3/dense_26/BiasAdd/ReadVariableOp'^model_3/dense_26/MatMul/ReadVariableOp(^model_3/dense_27/BiasAdd/ReadVariableOp'^model_3/dense_27/MatMul/ReadVariableOpD^model_3/token_and_position_embedding_3/embedding_6/embedding_lookupD^model_3/token_and_position_embedding_3/embedding_7/embedding_lookupL^model_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpP^model_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpL^model_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpP^model_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpW^model_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpa^model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpJ^model_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpT^model_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpL^model_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpV^model_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpL^model_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpV^model_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpI^model_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpK^model_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpI^model_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpK^model_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2p
6model_3/batch_normalization_6/batchnorm/ReadVariableOp6model_3/batch_normalization_6/batchnorm/ReadVariableOp2t
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_18model_3/batch_normalization_6/batchnorm/ReadVariableOp_12t
8model_3/batch_normalization_6/batchnorm/ReadVariableOp_28model_3/batch_normalization_6/batchnorm/ReadVariableOp_22x
:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp:model_3/batch_normalization_6/batchnorm/mul/ReadVariableOp2p
6model_3/batch_normalization_7/batchnorm/ReadVariableOp6model_3/batch_normalization_7/batchnorm/ReadVariableOp2t
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_18model_3/batch_normalization_7/batchnorm/ReadVariableOp_12t
8model_3/batch_normalization_7/batchnorm/ReadVariableOp_28model_3/batch_normalization_7/batchnorm/ReadVariableOp_22x
:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp:model_3/batch_normalization_7/batchnorm/mul/ReadVariableOp2R
'model_3/conv1d_6/BiasAdd/ReadVariableOp'model_3/conv1d_6/BiasAdd/ReadVariableOp2j
3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp3model_3/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2R
'model_3/conv1d_7/BiasAdd/ReadVariableOp'model_3/conv1d_7/BiasAdd/ReadVariableOp2j
3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp3model_3/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2R
'model_3/dense_25/BiasAdd/ReadVariableOp'model_3/dense_25/BiasAdd/ReadVariableOp2P
&model_3/dense_25/MatMul/ReadVariableOp&model_3/dense_25/MatMul/ReadVariableOp2R
'model_3/dense_26/BiasAdd/ReadVariableOp'model_3/dense_26/BiasAdd/ReadVariableOp2P
&model_3/dense_26/MatMul/ReadVariableOp&model_3/dense_26/MatMul/ReadVariableOp2R
'model_3/dense_27/BiasAdd/ReadVariableOp'model_3/dense_27/BiasAdd/ReadVariableOp2P
&model_3/dense_27/MatMul/ReadVariableOp&model_3/dense_27/MatMul/ReadVariableOp2К
Cmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookupCmodel_3/token_and_position_embedding_3/embedding_6/embedding_lookup2К
Cmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookupCmodel_3/token_and_position_embedding_3/embedding_7/embedding_lookup2Ъ
Kmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpKmodel_3/transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2Ґ
Omodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpOmodel_3/transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2Ъ
Kmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpKmodel_3/transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2Ґ
Omodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpOmodel_3/transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2∞
Vmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpVmodel_3/transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2ƒ
`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp`model_3/transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2Ц
Imodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpImodel_3/transformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2™
Smodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpSmodel_3/transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2Ъ
Kmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpKmodel_3/transformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2Ѓ
Umodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpUmodel_3/transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2Ъ
Kmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpKmodel_3/transformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2Ѓ
Umodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpUmodel_3/transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2Ф
Hmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpHmodel_3/transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2Ш
Jmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpJmodel_3/transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2Ф
Hmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpHmodel_3/transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2Ш
Jmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpJmodel_3/transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:R N
)
_output_shapes
:€€€€€€€€€†Ь
!
_user_specified_name	input_7:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_8
Љ0
»
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409933

inputs
assignmovingavg_409908
assignmovingavg_1_409914)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409908*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409908AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409908*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_409914*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_409914AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/409914*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
—
г
D__inference_dense_24_layer_call_and_return_conditional_losses_409700

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2	
BiasAddЬ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€B@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B@
 
_user_specified_nameinputs
…
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_410553

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Т€
в
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_410243

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_14/batchnorm/ReadVariableOpҐ3layer_normalization_14/batchnorm/mul/ReadVariableOpҐ/layer_normalization_15/batchnorm/ReadVariableOpҐ3layer_normalization_15/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_7/attention_output/add/ReadVariableOpҐDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_7/key/add/ReadVariableOpҐ7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/query/add/ReadVariableOpҐ9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/value/add/ReadVariableOpҐ9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐ,sequential_7/dense_23/BiasAdd/ReadVariableOpҐ.sequential_7/dense_23/Tensordot/ReadVariableOpҐ,sequential_7/dense_24/BiasAdd/ReadVariableOpҐ.sequential_7/dense_24/Tensordot/ReadVariableOpэ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/Einsumџ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpх
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/query/addч
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/Einsum’
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpн
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2 
multi_head_attention_7/key/addэ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/Einsumџ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpх
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/value/addБ
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_7/Mul/y∆
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 2
multi_head_attention_7/Mulь
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/Einsumƒ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2(
&multi_head_attention_7/softmax/Softmax°
,multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_7/dropout/dropout/ConstВ
*multi_head_attention_7/dropout/dropout/MulMul0multi_head_attention_7/softmax/Softmax:softmax:05multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2,
*multi_head_attention_7/dropout/dropout/MulЉ
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/ShapeЩ
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB*
dtype02E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_7/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB25
3multi_head_attention_7/dropout/dropout/GreaterEqualд
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€BB2-
+multi_head_attention_7/dropout/dropout/Castю
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€BB2.
,multi_head_attention_7/dropout/dropout/Mul_1Ф
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/EinsumЮ
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumш
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOpЭ
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2-
+multi_head_attention_7/attention_output/addy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_20/dropout/ConstЅ
dropout_20/dropout/MulMul/multi_head_attention_7/attention_output/add:z:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/dropout/MulУ
dropout_20/dropout/ShapeShape/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shapeў
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
dtype021
/dropout_20/dropout/random_uniform/RandomUniformЛ
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_20/dropout/GreaterEqual/yо
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
dropout_20/dropout/GreaterEqual§
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/dropout/Cast™
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/dropout/Mul_1o
addAddV2inputsdropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
addЄ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesв
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_14/moments/meanќ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_14/moments/StopGradientо
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_14/moments/SquaredDifferenceј
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indicesЫ
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_14/moments/varianceХ
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_14/batchnorm/add/yо
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_14/batchnorm/addє
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_14/batchnorm/Rsqrtг
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpт
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/mulј
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_1е
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_2„
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpо
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/subе
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/add_1Ў
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOpЦ
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axesЭ
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free®
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape†
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axisњ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2§
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1Ш
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstЎ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/ProdЬ
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1а
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1Ь
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axisЮ
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatд
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackц
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2+
)sequential_7/dense_23/Tensordot/transposeч
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_23/Tensordot/Reshapeц
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_7/dense_23/Tensordot/MatMulЬ
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2†
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axisЂ
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1и
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2!
sequential_7/dense_23/Tensordotќ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpя
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/BiasAddЮ
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/ReluЎ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOpЦ
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axesЭ
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¶
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape†
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axisњ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2§
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1Ш
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstЎ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/ProdЬ
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1а
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1Ь
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axisЮ
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatд
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackф
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2+
)sequential_7/dense_24/Tensordot/transposeч
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_24/Tensordot/Reshapeц
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_7/dense_24/Tensordot/MatMulЬ
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2†
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axisЂ
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1и
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
sequential_7/dense_24/Tensordotќ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpя
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
sequential_7/dense_24/BiasAddy
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_21/dropout/ConstЄ
dropout_21/dropout/MulMul&sequential_7/dense_24/BiasAdd:output:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/dropout/MulК
dropout_21/dropout/ShapeShape&sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shapeў
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
dtype021
/dropout_21/dropout/random_uniform/RandomUniformЛ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_21/dropout/GreaterEqual/yо
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
dropout_21/dropout/GreaterEqual§
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/dropout/Cast™
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/dropout/Mul_1Ч
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
add_1Є
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesд
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_15/moments/meanќ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_15/moments/StopGradientр
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_15/moments/SquaredDifferenceј
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indicesЫ
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_15/moments/varianceХ
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_15/batchnorm/add/yо
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_15/batchnorm/addє
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_15/batchnorm/Rsqrtг
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpт
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/mul¬
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_1е
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_2„
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpо
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/subе
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/add_1№
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€B ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
й
Б
H__inference_sequential_7_layer_call_and_return_conditional_losses_409748

inputs
dense_23_409737
dense_23_409739
dense_24_409742
dense_24_409744
identityИҐ dense_23/StatefulPartitionedCallҐ dense_24/StatefulPartitionedCallЫ
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_409737dense_23_409739*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCallЊ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409742dense_24_409744*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
 dense_24/StatefulPartitionedCall«
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
ѓ£
ю(
__inference__traced_save_413258
file_prefix.
*savev2_conv1d_6_kernel_read_readvariableop,
(savev2_conv1d_6_bias_read_readvariableop.
*savev2_conv1d_7_kernel_read_readvariableop,
(savev2_conv1d_7_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	T
Psavev2_token_and_position_embedding_3_embedding_6_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_3_embedding_7_embeddings_read_readvariableopV
Rsavev2_transformer_block_7_multi_head_attention_7_query_kernel_read_readvariableopT
Psavev2_transformer_block_7_multi_head_attention_7_query_bias_read_readvariableopT
Psavev2_transformer_block_7_multi_head_attention_7_key_kernel_read_readvariableopR
Nsavev2_transformer_block_7_multi_head_attention_7_key_bias_read_readvariableopV
Rsavev2_transformer_block_7_multi_head_attention_7_value_kernel_read_readvariableopT
Psavev2_transformer_block_7_multi_head_attention_7_value_bias_read_readvariableopa
]savev2_transformer_block_7_multi_head_attention_7_attention_output_kernel_read_readvariableop_
[savev2_transformer_block_7_multi_head_attention_7_attention_output_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableopO
Ksavev2_transformer_block_7_layer_normalization_14_gamma_read_readvariableopN
Jsavev2_transformer_block_7_layer_normalization_14_beta_read_readvariableopO
Ksavev2_transformer_block_7_layer_normalization_15_gamma_read_readvariableopN
Jsavev2_transformer_block_7_layer_normalization_15_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_sgd_conv1d_6_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_6_bias_momentum_read_readvariableop;
7savev2_sgd_conv1d_7_kernel_momentum_read_readvariableop9
5savev2_sgd_conv1d_7_bias_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopG
Csavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopF
Bsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableop;
7savev2_sgd_dense_25_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_25_bias_momentum_read_readvariableop;
7savev2_sgd_dense_26_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_26_bias_momentum_read_readvariableop;
7savev2_sgd_dense_27_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_27_bias_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentum_read_readvariableopa
]savev2_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum_read_readvariableopc
_savev2_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentum_read_readvariableopa
]savev2_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentum_read_readvariableop_
[savev2_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum_read_readvariableopc
_savev2_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentum_read_readvariableopa
]savev2_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentum_read_readvariableopn
jsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentum_read_readvariableopl
hsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum_read_readvariableop;
7savev2_sgd_dense_23_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_23_bias_momentum_read_readvariableop;
7savev2_sgd_dense_24_kernel_momentum_read_readvariableop9
5savev2_sgd_dense_24_bias_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_7_layer_normalization_14_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_7_layer_normalization_14_beta_momentum_read_readvariableop\
Xsavev2_sgd_transformer_block_7_layer_normalization_15_gamma_momentum_read_readvariableop[
Wsavev2_sgd_transformer_block_7_layer_normalization_15_beta_momentum_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЋ(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ё'
value”'B–'KB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBWlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBStrainable_variables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBTtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names°
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*Ђ
value°BЮKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesо'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_6_kernel_read_readvariableop(savev2_conv1d_6_bias_read_readvariableop*savev2_conv1d_7_kernel_read_readvariableop(savev2_conv1d_7_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop#savev2_sgd_iter_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_6_embeddings_read_readvariableopPsavev2_token_and_position_embedding_3_embedding_7_embeddings_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_query_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_query_bias_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_key_kernel_read_readvariableopNsavev2_transformer_block_7_multi_head_attention_7_key_bias_read_readvariableopRsavev2_transformer_block_7_multi_head_attention_7_value_kernel_read_readvariableopPsavev2_transformer_block_7_multi_head_attention_7_value_bias_read_readvariableop]savev2_transformer_block_7_multi_head_attention_7_attention_output_kernel_read_readvariableop[savev2_transformer_block_7_multi_head_attention_7_attention_output_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableopKsavev2_transformer_block_7_layer_normalization_14_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_14_beta_read_readvariableopKsavev2_transformer_block_7_layer_normalization_15_gamma_read_readvariableopJsavev2_transformer_block_7_layer_normalization_15_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_sgd_conv1d_6_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_6_bias_momentum_read_readvariableop7savev2_sgd_conv1d_7_kernel_momentum_read_readvariableop5savev2_sgd_conv1d_7_bias_momentum_read_readvariableopCsavev2_sgd_batch_normalization_6_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_6_beta_momentum_read_readvariableopCsavev2_sgd_batch_normalization_7_gamma_momentum_read_readvariableopBsavev2_sgd_batch_normalization_7_beta_momentum_read_readvariableop7savev2_sgd_dense_25_kernel_momentum_read_readvariableop5savev2_sgd_dense_25_bias_momentum_read_readvariableop7savev2_sgd_dense_26_kernel_momentum_read_readvariableop5savev2_sgd_dense_26_bias_momentum_read_readvariableop7savev2_sgd_dense_27_kernel_momentum_read_readvariableop5savev2_sgd_dense_27_bias_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_3_embedding_6_embeddings_momentum_read_readvariableop]savev2_sgd_token_and_position_embedding_3_embedding_7_embeddings_momentum_read_readvariableop_savev2_sgd_transformer_block_7_multi_head_attention_7_query_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_query_bias_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_key_kernel_momentum_read_readvariableop[savev2_sgd_transformer_block_7_multi_head_attention_7_key_bias_momentum_read_readvariableop_savev2_sgd_transformer_block_7_multi_head_attention_7_value_kernel_momentum_read_readvariableop]savev2_sgd_transformer_block_7_multi_head_attention_7_value_bias_momentum_read_readvariableopjsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_kernel_momentum_read_readvariableophsavev2_sgd_transformer_block_7_multi_head_attention_7_attention_output_bias_momentum_read_readvariableop7savev2_sgd_dense_23_kernel_momentum_read_readvariableop5savev2_sgd_dense_23_bias_momentum_read_readvariableop7savev2_sgd_dense_24_kernel_momentum_read_readvariableop5savev2_sgd_dense_24_bias_momentum_read_readvariableopXsavev2_sgd_transformer_block_7_layer_normalization_14_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_7_layer_normalization_14_beta_momentum_read_readvariableopXsavev2_sgd_transformer_block_7_layer_normalization_15_gamma_momentum_read_readvariableopWsavev2_sgd_transformer_block_7_layer_normalization_15_beta_momentum_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*у
_input_shapesб
ё: :  : :	  : : : : : : : : : :	»@:@:@@:@:@:: : : : : :
†Ь :  : :  : :  : :  : : @:@:@ : : : : : : : :  : :	  : : : : : :	»@:@:@@:@:@:: :
†Ь :  : :  : :  : :  : : @:@:@ : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:	  : 
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
:	»@: 
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

: :&"
 
_output_shapes
:
†Ь :($
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
: :(+$
"
_output_shapes
:  : ,

_output_shapes
: :(-$
"
_output_shapes
:	  : .
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
:	»@: 4
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

: :&:"
 
_output_shapes
:
†Ь :(;$
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
Ј
k
A__inference_add_3_layer_call_and_return_conditional_losses_410086

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:€€€€€€€€€B 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:€€€€€€€€€B :€€€€€€€€€B :S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
х
k
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_409303

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDimsЇ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
AvgPoolО
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Б
Й
H__inference_sequential_7_layer_call_and_return_conditional_losses_409717
dense_23_input
dense_23_409665
dense_23_409667
dense_24_409711
dense_24_409713
identityИҐ dense_23/StatefulPartitionedCallҐ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_409665dense_23_409667*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCallЊ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409711dense_24_409713*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
 dense_24/StatefulPartitionedCall«
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€B 
(
_user_specified_namedense_23_input
Љ0
»
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_410024

inputs
assignmovingavg_409999
assignmovingavg_1_410005)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_409999*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_409999AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/409999*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_410005*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_410005AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/410005*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
–

а
4__inference_transformer_block_7_layer_call_fn_412619

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
identityИҐStatefulPartitionedCallЅ
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
:€€€€€€€€€B *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4102432
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€B ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Є
†
-__inference_sequential_7_layer_call_fn_412933

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097752
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
и
И
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_410044

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
п
~
)__inference_dense_23_layer_call_fn_412973

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€B ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
»
©
6__inference_batch_normalization_7_layer_call_fn_412282

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100242
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Д
P
4__inference_average_pooling1d_9_layer_call_fn_409309

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
~
)__inference_dense_26_layer_call_fn_412747

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Љ0
»
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412003

inputs
assignmovingavg_411978
assignmovingavg_1_411984)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_411978*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_411978AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/411978*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_411984*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_411984AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/411984*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
•
d
+__inference_dropout_23_layer_call_fn_412769

inputs
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ь
Д
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_409815
x'
#embedding_7_embedding_lookup_409802'
#embedding_6_embedding_lookup_409808
identityИҐembedding_6/embedding_lookupҐembedding_7/embedding_lookup?
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
€€€€€€€€€2
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
strided_slice/stack_2в
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
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
rangeѓ
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_409802range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/409802*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_7/embedding_lookupЩ
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/409802*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_7/embedding_lookup/Identityј
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_7/embedding_lookup/Identity_1r
embedding_6/CastCastx*

DstT0*

SrcT0*)
_output_shapes
:€€€€€€€€€†Ь2
embedding_6/Castї
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_409808embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/409808*-
_output_shapes
:€€€€€€€€€†Ь *
dtype02
embedding_6/embedding_lookupЯ
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/409808*-
_output_shapes
:€€€€€€€€€†Ь 2'
%embedding_6/embedding_lookup/Identity∆
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2)
'embedding_6/embedding_lookup/Identity_1ѓ
addAddV20embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
addЯ
IdentityIdentityadd:z:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€†Ь::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup:L H
)
_output_shapes
:€€€€€€€€€†Ь

_user_specified_namex
Т€
в
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412455

inputsF
Bmulti_head_attention_7_query_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_query_add_readvariableop_resourceD
@multi_head_attention_7_key_einsum_einsum_readvariableop_resource:
6multi_head_attention_7_key_add_readvariableop_resourceF
Bmulti_head_attention_7_value_einsum_einsum_readvariableop_resource<
8multi_head_attention_7_value_add_readvariableop_resourceQ
Mmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resourceG
Cmulti_head_attention_7_attention_output_add_readvariableop_resource@
<layer_normalization_14_batchnorm_mul_readvariableop_resource<
8layer_normalization_14_batchnorm_readvariableop_resource;
7sequential_7_dense_23_tensordot_readvariableop_resource9
5sequential_7_dense_23_biasadd_readvariableop_resource;
7sequential_7_dense_24_tensordot_readvariableop_resource9
5sequential_7_dense_24_biasadd_readvariableop_resource@
<layer_normalization_15_batchnorm_mul_readvariableop_resource<
8layer_normalization_15_batchnorm_readvariableop_resource
identityИҐ/layer_normalization_14/batchnorm/ReadVariableOpҐ3layer_normalization_14/batchnorm/mul/ReadVariableOpҐ/layer_normalization_15/batchnorm/ReadVariableOpҐ3layer_normalization_15/batchnorm/mul/ReadVariableOpҐ:multi_head_attention_7/attention_output/add/ReadVariableOpҐDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_7/key/add/ReadVariableOpҐ7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/query/add/ReadVariableOpҐ9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_7/value/add/ReadVariableOpҐ9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐ,sequential_7/dense_23/BiasAdd/ReadVariableOpҐ.sequential_7/dense_23/Tensordot/ReadVariableOpҐ,sequential_7/dense_24/BiasAdd/ReadVariableOpҐ.sequential_7/dense_24/Tensordot/ReadVariableOpэ
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/query/einsum/EinsumEinsuminputsAmulti_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/query/einsum/Einsumџ
/multi_head_attention_7/query/add/ReadVariableOpReadVariableOp8multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/query/add/ReadVariableOpх
 multi_head_attention_7/query/addAddV23multi_head_attention_7/query/einsum/Einsum:output:07multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/query/addч
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype029
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOpЗ
(multi_head_attention_7/key/einsum/EinsumEinsuminputs?multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2*
(multi_head_attention_7/key/einsum/Einsum’
-multi_head_attention_7/key/add/ReadVariableOpReadVariableOp6multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02/
-multi_head_attention_7/key/add/ReadVariableOpн
multi_head_attention_7/key/addAddV21multi_head_attention_7/key/einsum/Einsum:output:05multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2 
multi_head_attention_7/key/addэ
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02;
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOpН
*multi_head_attention_7/value/einsum/EinsumEinsuminputsAmulti_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2,
*multi_head_attention_7/value/einsum/Einsumџ
/multi_head_attention_7/value/add/ReadVariableOpReadVariableOp8multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype021
/multi_head_attention_7/value/add/ReadVariableOpх
 multi_head_attention_7/value/addAddV23multi_head_attention_7/value/einsum/Einsum:output:07multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 2"
 multi_head_attention_7/value/addБ
multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>2
multi_head_attention_7/Mul/y∆
multi_head_attention_7/MulMul$multi_head_attention_7/query/add:z:0%multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 2
multi_head_attention_7/Mulь
$multi_head_attention_7/einsum/EinsumEinsum"multi_head_attention_7/key/add:z:0multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2&
$multi_head_attention_7/einsum/Einsumƒ
&multi_head_attention_7/softmax/SoftmaxSoftmax-multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2(
&multi_head_attention_7/softmax/Softmax°
,multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,multi_head_attention_7/dropout/dropout/ConstВ
*multi_head_attention_7/dropout/dropout/MulMul0multi_head_attention_7/softmax/Softmax:softmax:05multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2,
*multi_head_attention_7/dropout/dropout/MulЉ
,multi_head_attention_7/dropout/dropout/ShapeShape0multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_7/dropout/dropout/ShapeЩ
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB*
dtype02E
Cmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5multi_head_attention_7/dropout/dropout/GreaterEqual/y¬
3multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB25
3multi_head_attention_7/dropout/dropout/GreaterEqualд
+multi_head_attention_7/dropout/dropout/CastCast7multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€BB2-
+multi_head_attention_7/dropout/dropout/Castю
,multi_head_attention_7/dropout/dropout/Mul_1Mul.multi_head_attention_7/dropout/dropout/Mul:z:0/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€BB2.
,multi_head_attention_7/dropout/dropout/Mul_1Ф
&multi_head_attention_7/einsum_1/EinsumEinsum0multi_head_attention_7/dropout/dropout/Mul_1:z:0$multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2(
&multi_head_attention_7/einsum_1/EinsumЮ
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02F
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp”
5multi_head_attention_7/attention_output/einsum/EinsumEinsum/multi_head_attention_7/einsum_1/Einsum:output:0Lmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe27
5multi_head_attention_7/attention_output/einsum/Einsumш
:multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02<
:multi_head_attention_7/attention_output/add/ReadVariableOpЭ
+multi_head_attention_7/attention_output/addAddV2>multi_head_attention_7/attention_output/einsum/Einsum:output:0Bmulti_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2-
+multi_head_attention_7/attention_output/addy
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_20/dropout/ConstЅ
dropout_20/dropout/MulMul/multi_head_attention_7/attention_output/add:z:0!dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/dropout/MulУ
dropout_20/dropout/ShapeShape/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shapeў
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
dtype021
/dropout_20/dropout/random_uniform/RandomUniformЛ
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_20/dropout/GreaterEqual/yо
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
dropout_20/dropout/GreaterEqual§
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/dropout/Cast™
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_20/dropout/Mul_1o
addAddV2inputsdropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
addЄ
5layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_14/moments/mean/reduction_indicesв
#layer_normalization_14/moments/meanMeanadd:z:0>layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_14/moments/meanќ
+layer_normalization_14/moments/StopGradientStopGradient,layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_14/moments/StopGradientо
0layer_normalization_14/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_14/moments/SquaredDifferenceј
9layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_14/moments/variance/reduction_indicesЫ
'layer_normalization_14/moments/varianceMean4layer_normalization_14/moments/SquaredDifference:z:0Blayer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_14/moments/varianceХ
&layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_14/batchnorm/add/yо
$layer_normalization_14/batchnorm/addAddV20layer_normalization_14/moments/variance:output:0/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_14/batchnorm/addє
&layer_normalization_14/batchnorm/RsqrtRsqrt(layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_14/batchnorm/Rsqrtг
3layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_14/batchnorm/mul/ReadVariableOpт
$layer_normalization_14/batchnorm/mulMul*layer_normalization_14/batchnorm/Rsqrt:y:0;layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/mulј
&layer_normalization_14/batchnorm/mul_1Muladd:z:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_1е
&layer_normalization_14/batchnorm/mul_2Mul,layer_normalization_14/moments/mean:output:0(layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/mul_2„
/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_14/batchnorm/ReadVariableOpо
$layer_normalization_14/batchnorm/subSub7layer_normalization_14/batchnorm/ReadVariableOp:value:0*layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_14/batchnorm/subе
&layer_normalization_14/batchnorm/add_1AddV2*layer_normalization_14/batchnorm/mul_1:z:0(layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_14/batchnorm/add_1Ў
.sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype020
.sequential_7/dense_23/Tensordot/ReadVariableOpЦ
$sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_23/Tensordot/axesЭ
$sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_23/Tensordot/free®
%sequential_7/dense_23/Tensordot/ShapeShape*layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/Shape†
-sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/GatherV2/axisњ
(sequential_7/dense_23/Tensordot/GatherV2GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/free:output:06sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/GatherV2§
/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_23/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_23/Tensordot/GatherV2_1GatherV2.sequential_7/dense_23/Tensordot/Shape:output:0-sequential_7/dense_23/Tensordot/axes:output:08sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_23/Tensordot/GatherV2_1Ш
%sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_23/Tensordot/ConstЎ
$sequential_7/dense_23/Tensordot/ProdProd1sequential_7/dense_23/Tensordot/GatherV2:output:0.sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_23/Tensordot/ProdЬ
'sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_23/Tensordot/Const_1а
&sequential_7/dense_23/Tensordot/Prod_1Prod3sequential_7/dense_23/Tensordot/GatherV2_1:output:00sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_23/Tensordot/Prod_1Ь
+sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_23/Tensordot/concat/axisЮ
&sequential_7/dense_23/Tensordot/concatConcatV2-sequential_7/dense_23/Tensordot/free:output:0-sequential_7/dense_23/Tensordot/axes:output:04sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_23/Tensordot/concatд
%sequential_7/dense_23/Tensordot/stackPack-sequential_7/dense_23/Tensordot/Prod:output:0/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_23/Tensordot/stackц
)sequential_7/dense_23/Tensordot/transpose	Transpose*layer_normalization_14/batchnorm/add_1:z:0/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2+
)sequential_7/dense_23/Tensordot/transposeч
'sequential_7/dense_23/Tensordot/ReshapeReshape-sequential_7/dense_23/Tensordot/transpose:y:0.sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_23/Tensordot/Reshapeц
&sequential_7/dense_23/Tensordot/MatMulMatMul0sequential_7/dense_23/Tensordot/Reshape:output:06sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_7/dense_23/Tensordot/MatMulЬ
'sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2)
'sequential_7/dense_23/Tensordot/Const_2†
-sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_23/Tensordot/concat_1/axisЂ
(sequential_7/dense_23/Tensordot/concat_1ConcatV21sequential_7/dense_23/Tensordot/GatherV2:output:00sequential_7/dense_23/Tensordot/Const_2:output:06sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_23/Tensordot/concat_1и
sequential_7/dense_23/TensordotReshape0sequential_7/dense_23/Tensordot/MatMul:product:01sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2!
sequential_7/dense_23/Tensordotќ
,sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_7/dense_23/BiasAdd/ReadVariableOpя
sequential_7/dense_23/BiasAddBiasAdd(sequential_7/dense_23/Tensordot:output:04sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/BiasAddЮ
sequential_7/dense_23/ReluRelu&sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
sequential_7/dense_23/ReluЎ
.sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOp7sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype020
.sequential_7/dense_24/Tensordot/ReadVariableOpЦ
$sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_7/dense_24/Tensordot/axesЭ
$sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_7/dense_24/Tensordot/free¶
%sequential_7/dense_24/Tensordot/ShapeShape(sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/Shape†
-sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/GatherV2/axisњ
(sequential_7/dense_24/Tensordot/GatherV2GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/free:output:06sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/GatherV2§
/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_7/dense_24/Tensordot/GatherV2_1/axis≈
*sequential_7/dense_24/Tensordot/GatherV2_1GatherV2.sequential_7/dense_24/Tensordot/Shape:output:0-sequential_7/dense_24/Tensordot/axes:output:08sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_7/dense_24/Tensordot/GatherV2_1Ш
%sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_7/dense_24/Tensordot/ConstЎ
$sequential_7/dense_24/Tensordot/ProdProd1sequential_7/dense_24/Tensordot/GatherV2:output:0.sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_7/dense_24/Tensordot/ProdЬ
'sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_1а
&sequential_7/dense_24/Tensordot/Prod_1Prod3sequential_7/dense_24/Tensordot/GatherV2_1:output:00sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_7/dense_24/Tensordot/Prod_1Ь
+sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_7/dense_24/Tensordot/concat/axisЮ
&sequential_7/dense_24/Tensordot/concatConcatV2-sequential_7/dense_24/Tensordot/free:output:0-sequential_7/dense_24/Tensordot/axes:output:04sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_7/dense_24/Tensordot/concatд
%sequential_7/dense_24/Tensordot/stackPack-sequential_7/dense_24/Tensordot/Prod:output:0/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_24/Tensordot/stackф
)sequential_7/dense_24/Tensordot/transpose	Transpose(sequential_7/dense_23/Relu:activations:0/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2+
)sequential_7/dense_24/Tensordot/transposeч
'sequential_7/dense_24/Tensordot/ReshapeReshape-sequential_7/dense_24/Tensordot/transpose:y:0.sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2)
'sequential_7/dense_24/Tensordot/Reshapeц
&sequential_7/dense_24/Tensordot/MatMulMatMul0sequential_7/dense_24/Tensordot/Reshape:output:06sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_7/dense_24/Tensordot/MatMulЬ
'sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_7/dense_24/Tensordot/Const_2†
-sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_7/dense_24/Tensordot/concat_1/axisЂ
(sequential_7/dense_24/Tensordot/concat_1ConcatV21sequential_7/dense_24/Tensordot/GatherV2:output:00sequential_7/dense_24/Tensordot/Const_2:output:06sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_7/dense_24/Tensordot/concat_1и
sequential_7/dense_24/TensordotReshape0sequential_7/dense_24/Tensordot/MatMul:product:01sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
sequential_7/dense_24/Tensordotќ
,sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_7/dense_24/BiasAdd/ReadVariableOpя
sequential_7/dense_24/BiasAddBiasAdd(sequential_7/dense_24/Tensordot:output:04sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
sequential_7/dense_24/BiasAddy
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_21/dropout/ConstЄ
dropout_21/dropout/MulMul&sequential_7/dense_24/BiasAdd:output:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/dropout/MulК
dropout_21/dropout/ShapeShape&sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shapeў
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
dtype021
/dropout_21/dropout/random_uniform/RandomUniformЛ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_21/dropout/GreaterEqual/yо
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2!
dropout_21/dropout/GreaterEqual§
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/dropout/Cast™
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dropout_21/dropout/Mul_1Ч
add_1AddV2*layer_normalization_14/batchnorm/add_1:z:0dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
add_1Є
5layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_15/moments/mean/reduction_indicesд
#layer_normalization_15/moments/meanMean	add_1:z:0>layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2%
#layer_normalization_15/moments/meanќ
+layer_normalization_15/moments/StopGradientStopGradient,layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2-
+layer_normalization_15/moments/StopGradientр
0layer_normalization_15/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 22
0layer_normalization_15/moments/SquaredDifferenceј
9layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_15/moments/variance/reduction_indicesЫ
'layer_normalization_15/moments/varianceMean4layer_normalization_15/moments/SquaredDifference:z:0Blayer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2)
'layer_normalization_15/moments/varianceХ
&layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52(
&layer_normalization_15/batchnorm/add/yо
$layer_normalization_15/batchnorm/addAddV20layer_normalization_15/moments/variance:output:0/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2&
$layer_normalization_15/batchnorm/addє
&layer_normalization_15/batchnorm/RsqrtRsqrt(layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2(
&layer_normalization_15/batchnorm/Rsqrtг
3layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_15/batchnorm/mul/ReadVariableOpт
$layer_normalization_15/batchnorm/mulMul*layer_normalization_15/batchnorm/Rsqrt:y:0;layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/mul¬
&layer_normalization_15/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_1е
&layer_normalization_15/batchnorm/mul_2Mul,layer_normalization_15/moments/mean:output:0(layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/mul_2„
/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_15/batchnorm/ReadVariableOpо
$layer_normalization_15/batchnorm/subSub7layer_normalization_15/batchnorm/ReadVariableOp:value:0*layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2&
$layer_normalization_15/batchnorm/subе
&layer_normalization_15/batchnorm/add_1AddV2*layer_normalization_15/batchnorm/mul_1:z:0(layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2(
&layer_normalization_15/batchnorm/add_1№
IdentityIdentity*layer_normalization_15/batchnorm/add_1:z:00^layer_normalization_14/batchnorm/ReadVariableOp4^layer_normalization_14/batchnorm/mul/ReadVariableOp0^layer_normalization_15/batchnorm/ReadVariableOp4^layer_normalization_15/batchnorm/mul/ReadVariableOp;^multi_head_attention_7/attention_output/add/ReadVariableOpE^multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_7/key/add/ReadVariableOp8^multi_head_attention_7/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/query/add/ReadVariableOp:^multi_head_attention_7/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_7/value/add/ReadVariableOp:^multi_head_attention_7/value/einsum/Einsum/ReadVariableOp-^sequential_7/dense_23/BiasAdd/ReadVariableOp/^sequential_7/dense_23/Tensordot/ReadVariableOp-^sequential_7/dense_24/BiasAdd/ReadVariableOp/^sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€B ::::::::::::::::2b
/layer_normalization_14/batchnorm/ReadVariableOp/layer_normalization_14/batchnorm/ReadVariableOp2j
3layer_normalization_14/batchnorm/mul/ReadVariableOp3layer_normalization_14/batchnorm/mul/ReadVariableOp2b
/layer_normalization_15/batchnorm/ReadVariableOp/layer_normalization_15/batchnorm/ReadVariableOp2j
3layer_normalization_15/batchnorm/mul/ReadVariableOp3layer_normalization_15/batchnorm/mul/ReadVariableOp2x
:multi_head_attention_7/attention_output/add/ReadVariableOp:multi_head_attention_7/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_7/key/add/ReadVariableOp-multi_head_attention_7/key/add/ReadVariableOp2r
7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp7multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/query/add/ReadVariableOp/multi_head_attention_7/query/add/ReadVariableOp2v
9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp9multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_7/value/add/ReadVariableOp/multi_head_attention_7/value/add/ReadVariableOp2v
9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp9multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2\
,sequential_7/dense_23/BiasAdd/ReadVariableOp,sequential_7/dense_23/BiasAdd/ReadVariableOp2`
.sequential_7/dense_23/Tensordot/ReadVariableOp.sequential_7/dense_23/Tensordot/ReadVariableOp2\
,sequential_7/dense_24/BiasAdd/ReadVariableOp,sequential_7/dense_24/BiasAdd/ReadVariableOp2`
.sequential_7/dense_24/Tensordot/ReadVariableOp.sequential_7/dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
 
©
6__inference_batch_normalization_7_layer_call_fn_412295

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallҐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100442
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
ц
l
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_409318

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimУ

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDimsЇ
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize

*
paddingVALID*
strides

2	
AvgPoolО
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
й
Б
H__inference_sequential_7_layer_call_and_return_conditional_losses_409775

inputs
dense_23_409764
dense_23_409766
dense_24_409769
dense_24_409771
identityИҐ dense_23/StatefulPartitionedCallҐ dense_24/StatefulPartitionedCallЫ
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_409764dense_23_409766*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCallЊ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409769dense_24_409771*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
 dense_24/StatefulPartitionedCall«
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
ь
Д
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_411908
x'
#embedding_7_embedding_lookup_411895'
#embedding_6_embedding_lookup_411901
identityИҐembedding_6/embedding_lookupҐembedding_7/embedding_lookup?
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
€€€€€€€€€2
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
strided_slice/stack_2в
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
range/deltaА
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:€€€€€€€€€2
rangeѓ
embedding_7/embedding_lookupResourceGather#embedding_7_embedding_lookup_411895range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_7/embedding_lookup/411895*'
_output_shapes
:€€€€€€€€€ *
dtype02
embedding_7/embedding_lookupЩ
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_7/embedding_lookup/411895*'
_output_shapes
:€€€€€€€€€ 2'
%embedding_7/embedding_lookup/Identityј
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'embedding_7/embedding_lookup/Identity_1r
embedding_6/CastCastx*

DstT0*

SrcT0*)
_output_shapes
:€€€€€€€€€†Ь2
embedding_6/Castї
embedding_6/embedding_lookupResourceGather#embedding_6_embedding_lookup_411901embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_6/embedding_lookup/411901*-
_output_shapes
:€€€€€€€€€†Ь *
dtype02
embedding_6/embedding_lookupЯ
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_6/embedding_lookup/411901*-
_output_shapes
:€€€€€€€€€†Ь 2'
%embedding_6/embedding_lookup/Identity∆
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2)
'embedding_6/embedding_lookup/Identity_1ѓ
addAddV20embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
addЯ
IdentityIdentityadd:z:0^embedding_6/embedding_lookup^embedding_7/embedding_lookup*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€†Ь::2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2<
embedding_7/embedding_lookupembedding_7/embedding_lookup:L H
)
_output_shapes
:€€€€€€€€€†Ь

_user_specified_namex
µ
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_410485

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€B :S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
п
~
)__inference_dense_24_layer_call_fn_413012

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€B@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B@
 
_user_specified_nameinputs
у
~
)__inference_conv1d_7_layer_call_fn_411967

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€Ъ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Ъ 
 
_user_specified_nameinputs
о	
Ё
D__inference_dense_26_layer_call_and_return_conditional_losses_410577

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
»
©
6__inference_batch_normalization_6_layer_call_fn_412036

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099332
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
о
©
6__inference_batch_normalization_7_layer_call_fn_412213

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4096082
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
њ
m
A__inference_add_3_layer_call_and_return_conditional_losses_412301
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:€€€€€€€€€B 2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:€€€€€€€€€B :€€€€€€€€€B :U Q
+
_output_shapes
:€€€€€€€€€B 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€B 
"
_user_specified_name
inputs/1
НJ
ѓ
H__inference_sequential_7_layer_call_and_return_conditional_losses_412850

inputs.
*dense_23_tensordot_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identityИҐdense_23/BiasAdd/ReadVariableOpҐ!dense_23/Tensordot/ReadVariableOpҐdense_24/BiasAdd/ReadVariableOpҐ!dense_24/Tensordot/ReadVariableOp±
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axesГ
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/freej
dense_23/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_23/Tensordot/ShapeЖ
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axisю
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2К
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axisД
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const§
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/ProdВ
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1ђ
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1В
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axisЁ
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat∞
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stackЂ
dense_23/Tensordot/transpose	Transposeinputs"dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dense_23/Tensordot/transpose√
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_23/Tensordot/Reshape¬
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_23/Tensordot/MatMulВ
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_23/Tensordot/Const_2Ж
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axisк
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1і
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_23/TensordotІ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOpЂ
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_23/BiasAddw
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_23/Relu±
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axesГ
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/free
dense_24/Tensordot/ShapeShapedense_23/Relu:activations:0*
T0*
_output_shapes
:2
dense_24/Tensordot/ShapeЖ
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axisю
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2К
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axisД
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2_1~
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const§
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/ProdВ
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1ђ
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1В
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axisЁ
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat∞
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stackј
dense_24/Tensordot/transpose	Transposedense_23/Relu:activations:0"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_24/Tensordot/transpose√
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_24/Tensordot/Reshape¬
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_24/Tensordot/MatMulВ
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_2Ж
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axisк
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1і
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dense_24/TensordotІ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOpЂ
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dense_24/BiasAddэ
IdentityIdentitydense_24/BiasAdd:output:0 ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
Ъ
ч
D__inference_conv1d_7_layer_call_and_return_conditional_losses_409880

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
paddingSAME*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
Relu©
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€Ъ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Ъ 
 
_user_specified_nameinputs
у0
»
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412167

inputs
assignmovingavg_412142
assignmovingavg_1_412148)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient±
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_412142*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_412142AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412142*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_412148*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_412148AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412148*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ў
§
(__inference_model_3_layer_call_fn_411806
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
identityИҐStatefulPartitionedCall“
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
:€€€€€€€€€*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4108422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
)
_output_shapes
:€€€€€€€€€†Ь
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
и
И
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412269

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1я
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
÷
Ґ
(__inference_model_3_layer_call_fn_411089
input_7
input_8
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
identityИҐStatefulPartitionedCall‘
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4110142
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:€€€€€€€€€†Ь
!
_user_specified_name	input_7:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_8
лX
Ю
C__inference_model_3_layer_call_and_return_conditional_losses_411014

inputs
inputs_1)
%token_and_position_embedding_3_410924)
%token_and_position_embedding_3_410926
conv1d_6_410929
conv1d_6_410931
conv1d_7_410935
conv1d_7_410937 
batch_normalization_6_410942 
batch_normalization_6_410944 
batch_normalization_6_410946 
batch_normalization_6_410948 
batch_normalization_7_410951 
batch_normalization_7_410953 
batch_normalization_7_410955 
batch_normalization_7_410957
transformer_block_7_410961
transformer_block_7_410963
transformer_block_7_410965
transformer_block_7_410967
transformer_block_7_410969
transformer_block_7_410971
transformer_block_7_410973
transformer_block_7_410975
transformer_block_7_410977
transformer_block_7_410979
transformer_block_7_410981
transformer_block_7_410983
transformer_block_7_410985
transformer_block_7_410987
transformer_block_7_410989
transformer_block_7_410991
dense_25_410996
dense_25_410998
dense_26_411002
dense_26_411004
dense_27_411008
dense_27_411010
identityИҐ-batch_normalization_6/StatefulPartitionedCallҐ-batch_normalization_7/StatefulPartitionedCallҐ conv1d_6/StatefulPartitionedCallҐ conv1d_7/StatefulPartitionedCallҐ dense_25/StatefulPartitionedCallҐ dense_26/StatefulPartitionedCallҐ dense_27/StatefulPartitionedCallҐ6token_and_position_embedding_3/StatefulPartitionedCallҐ+transformer_block_7/StatefulPartitionedCallЛ
6token_and_position_embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs%token_and_position_embedding_3_410924%token_and_position_embedding_3_410926*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_40981528
6token_and_position_embedding_3/StatefulPartitionedCall÷
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0conv1d_6_410929conv1d_6_410931*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_4098472"
 conv1d_6/StatefulPartitionedCall†
#average_pooling1d_9/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_4093032%
#average_pooling1d_9/PartitionedCall¬
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_9/PartitionedCall:output:0conv1d_7_410935conv1d_7_410937*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ъ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_4098802"
 conv1d_7/StatefulPartitionedCallЄ
$average_pooling1d_11/PartitionedCallPartitionedCall?token_and_position_embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_4093332&
$average_pooling1d_11/PartitionedCallҐ
$average_pooling1d_10/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_4093182&
$average_pooling1d_10/PartitionedCall√
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_10/PartitionedCall:output:0batch_normalization_6_410942batch_normalization_6_410944batch_normalization_6_410946batch_normalization_6_410948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4099532/
-batch_normalization_6/StatefulPartitionedCall√
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall-average_pooling1d_11/PartitionedCall:output:0batch_normalization_7_410951batch_normalization_7_410953batch_normalization_7_410955batch_normalization_7_410957*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4100442/
-batch_normalization_7/StatefulPartitionedCallї
add_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:06batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4100862
add_3/PartitionedCallО
+transformer_block_7/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0transformer_block_7_410961transformer_block_7_410963transformer_block_7_410965transformer_block_7_410967transformer_block_7_410969transformer_block_7_410971transformer_block_7_410973transformer_block_7_410975transformer_block_7_410977transformer_block_7_410979transformer_block_7_410981transformer_block_7_410983transformer_block_7_410985transformer_block_7_410987transformer_block_7_410989transformer_block_7_410991*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4103702-
+transformer_block_7/StatefulPartitionedCallЙ
flatten_3/PartitionedCallPartitionedCall4transformer_block_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_4104852
flatten_3/PartitionedCallО
concatenate_3/PartitionedCallPartitionedCall"flatten_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_3_layer_call_and_return_conditional_losses_4105002
concatenate_3/PartitionedCallЈ
 dense_25/StatefulPartitionedCallStatefulPartitionedCall&concatenate_3/PartitionedCall:output:0dense_25_410996dense_25_410998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_4105202"
 dense_25/StatefulPartitionedCallА
dropout_22/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105532
dropout_22/PartitionedCallі
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_26_411002dense_26_411004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_4105772"
 dense_26/StatefulPartitionedCallА
dropout_23/PartitionedCallPartitionedCall)dense_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_4106102
dropout_23/PartitionedCallі
 dense_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_27_411008dense_27_411010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332"
 dense_27/StatefulPartitionedCallу
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall7^token_and_position_embedding_3/StatefulPartitionedCall,^transformer_block_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2p
6token_and_position_embedding_3/StatefulPartitionedCall6token_and_position_embedding_3/StatefulPartitionedCall2Z
+transformer_block_7/StatefulPartitionedCall+transformer_block_7/StatefulPartitionedCall:Q M
)
_output_shapes
:€€€€€€€€€†Ь
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
о
©
6__inference_batch_normalization_6_layer_call_fn_412131

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4094682
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ц
И
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_409468

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpТ
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
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2
batchnorm/add_1и
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
‘№
ћ$
C__inference_model_3_layer_call_and_return_conditional_losses_411728
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_411497F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_4115038
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resourceZ
Vtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_query_add_readvariableop_resourceX
Ttransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resourceZ
Vtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_value_add_readvariableop_resourcee
atransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identityИҐ.batch_normalization_6/batchnorm/ReadVariableOpҐ0batch_normalization_6/batchnorm/ReadVariableOp_1Ґ0batch_normalization_6/batchnorm/ReadVariableOp_2Ґ2batch_normalization_6/batchnorm/mul/ReadVariableOpҐ.batch_normalization_7/batchnorm/ReadVariableOpҐ0batch_normalization_7/batchnorm/ReadVariableOp_1Ґ0batch_normalization_7/batchnorm/ReadVariableOp_2Ґ2batch_normalization_7/batchnorm/mul/ReadVariableOpҐconv1d_6/BiasAdd/ReadVariableOpҐ+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpҐconv1d_7/BiasAdd/ReadVariableOpҐ+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpҐdense_25/BiasAdd/ReadVariableOpҐdense_25/MatMul/ReadVariableOpҐdense_26/BiasAdd/ReadVariableOpҐdense_26/MatMul/ReadVariableOpҐdense_27/BiasAdd/ReadVariableOpҐdense_27/MatMul/ReadVariableOpҐ;token_and_position_embedding_3/embedding_6/embedding_lookupҐ;token_and_position_embedding_3/embedding_7/embedding_lookupҐCtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpҐGtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpҐCtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpҐGtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpҐNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpҐXtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpҐKtransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpҐMtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpҐMtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐ@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpҐBtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpҐ@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpҐBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpД
$token_and_position_embedding_3/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_3/Shapeї
2token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€24
2token_and_position_embedding_3/strided_slice/stackґ
4token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_3/strided_slice/stack_1ґ
4token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_3/strided_slice/stack_2Ь
,token_and_position_embedding_3/strided_sliceStridedSlice-token_and_position_embedding_3/Shape:output:0;token_and_position_embedding_3/strided_slice/stack:output:0=token_and_position_embedding_3/strided_slice/stack_1:output:0=token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_3/strided_sliceЪ
*token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_3/range/startЪ
*token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_3/range/deltaЫ
$token_and_position_embedding_3/rangeRange3token_and_position_embedding_3/range/start:output:05token_and_position_embedding_3/strided_slice:output:03token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2&
$token_and_position_embedding_3/range 
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_411497-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411497*'
_output_shapes
:€€€€€€€€€ *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookupХ
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411497*'
_output_shapes
:€€€€€€€€€ 2F
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityЭ
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2H
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1Ј
/token_and_position_embedding_3/embedding_6/CastCastinputs_0*

DstT0*

SrcT0*)
_output_shapes
:€€€€€€€€€†Ь21
/token_and_position_embedding_3/embedding_6/Cast÷
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_4115033token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411503*-
_output_shapes
:€€€€€€€€€†Ь *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookupЫ
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411503*-
_output_shapes
:€€€€€€€€€†Ь 2F
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity£
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2H
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1Ђ
"token_and_position_embedding_3/addAddV2Otoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2$
"token_and_position_embedding_3/addЛ
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_6/conv1d/ExpandDims/dim”
conv1d_6/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2
conv1d_6/conv1d/ExpandDims”
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dimџ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_6/conv1d/ExpandDims_1№
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь *
paddingSAME*
strides
2
conv1d_6/conv1dѓ
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь *
squeeze_dims

э€€€€€€€€2
conv1d_6/conv1d/SqueezeІ
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp≤
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
conv1d_6/BiasAddy
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
conv1d_6/ReluК
"average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_9/ExpandDims/dim‘
average_pooling1d_9/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_9/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2 
average_pooling1d_9/ExpandDimsе
average_pooling1d_9/AvgPoolAvgPool'average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_9/AvgPoolє
average_pooling1d_9/SqueezeSqueeze$average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims
2
average_pooling1d_9/SqueezeЛ
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_7/conv1d/ExpandDims/dim–
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_9/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2
conv1d_7/conv1d/ExpandDims”
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dimџ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_7/conv1d/ExpandDims_1џ
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
paddingSAME*
strides
2
conv1d_7/conv1dЃ
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims

э€€€€€€€€2
conv1d_7/conv1d/SqueezeІ
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp±
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
conv1d_7/BiasAddx
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
conv1d_7/ReluМ
#average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_11/ExpandDims/dimв
average_pooling1d_11/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0,average_pooling1d_11/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2!
average_pooling1d_11/ExpandDimsй
average_pooling1d_11/AvgPoolAvgPool(average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€B *
ksize	
ђ*
paddingVALID*
strides	
ђ2
average_pooling1d_11/AvgPoolї
average_pooling1d_11/SqueezeSqueeze%average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
squeeze_dims
2
average_pooling1d_11/SqueezeМ
#average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_10/ExpandDims/dim÷
average_pooling1d_10/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0,average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2!
average_pooling1d_10/ExpandDimsз
average_pooling1d_10/AvgPoolAvgPool(average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€B *
ksize

*
paddingVALID*
strides

2
average_pooling1d_10/AvgPoolї
average_pooling1d_10/SqueezeSqueeze%average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
squeeze_dims
2
average_pooling1d_10/Squeeze‘
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yа
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/add•
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/Rsqrtа
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/mulџ
%batch_normalization_6/batchnorm/mul_1Mul%average_pooling1d_10/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_6/batchnorm/mul_1Џ
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1Ё
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/mul_2Џ
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2џ
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/subб
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_6/batchnorm/add_1‘
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/yа
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/add•
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/Rsqrtа
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/mulџ
%batch_normalization_7/batchnorm/mul_1Mul%average_pooling1d_11/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_7/batchnorm/mul_1Џ
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1Ё
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/mul_2Џ
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2џ
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/subб
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_7/batchnorm/add_1Ђ
	add_3/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
	add_3/addє
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp–
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumЧ
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp≈
4transformer_block_7/multi_head_attention_7/query/addAddV2Gtransformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 26
4transformer_block_7/multi_head_attention_7/query/add≥
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp 
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsumadd_3/add:z:0Stransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2>
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumС
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpJtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpљ
2transformer_block_7/multi_head_attention_7/key/addAddV2Etransformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Itransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 24
2transformer_block_7/multi_head_attention_7/key/addє
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp–
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumЧ
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp≈
4transformer_block_7/multi_head_attention_7/value/addAddV2Gtransformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 26
4transformer_block_7/multi_head_attention_7/value/add©
0transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>22
0transformer_block_7/multi_head_attention_7/Mul/yЦ
.transformer_block_7/multi_head_attention_7/MulMul8transformer_block_7/multi_head_attention_7/query/add:z:09transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 20
.transformer_block_7/multi_head_attention_7/Mulћ
8transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum6transformer_block_7/multi_head_attention_7/key/add:z:02transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2:
8transformer_block_7/multi_head_attention_7/einsum/EinsumА
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxAtransformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2<
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxЖ
;transformer_block_7/multi_head_attention_7/dropout/IdentityIdentityDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€BB2=
;transformer_block_7/multi_head_attention_7/dropout/Identityд
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumDtransformer_block_7/multi_head_attention_7/dropout/Identity:output:08transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2<
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumЏ
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumCtransformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0`transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe2K
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsumі
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpн
?transformer_block_7/multi_head_attention_7/attention_output/addAddV2Rtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0Vtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2A
?transformer_block_7/multi_head_attention_7/attention_output/addў
'transformer_block_7/dropout_20/IdentityIdentityCtransformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2)
'transformer_block_7/dropout_20/Identity≤
transformer_block_7/addAddV2add_3/add:z:00transformer_block_7/dropout_20/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
transformer_block_7/addа
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indices≤
7transformer_block_7/layer_normalization_14/moments/meanMeantransformer_block_7/add:z:0Rtransformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(29
7transformer_block_7/layer_normalization_14/moments/meanК
?transformer_block_7/layer_normalization_14/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2A
?transformer_block_7/layer_normalization_14/moments/StopGradientЊ
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add:z:0Htransformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2F
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceи
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesл
;transformer_block_7/layer_normalization_14/moments/varianceMeanHtransformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2=
;transformer_block_7/layer_normalization_14/moments/varianceљ
:transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_7/layer_normalization_14/batchnorm/add/yЊ
8transformer_block_7/layer_normalization_14/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_14/moments/variance:output:0Ctransformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2:
8transformer_block_7/layer_normalization_14/batchnorm/addх
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2<
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtЯ
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp¬
8transformer_block_7/layer_normalization_14/batchnorm/mulMul>transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_14/batchnorm/mulР
:transformer_block_7/layer_normalization_14/batchnorm/mul_1Multransformer_block_7/add:z:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_14/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_14/moments/mean:output:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_2У
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpЊ
8transformer_block_7/layer_normalization_14/batchnorm/subSubKtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_14/batchnorm/subµ
:transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_14/batchnorm/add_1Ф
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpЊ
8transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_23/Tensordot/axes≈
8transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_23/Tensordot/freeд
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShape>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/Shape»
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2ћ
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1ј
9transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_23/Tensordot/Const®
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_23/Tensordot/Prodƒ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1∞
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ƒ
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisВ
:transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_23/Tensordot/concatі
9transformer_block_7/sequential_7/dense_23/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/stack∆
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose	Transpose>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2?
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose«
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Reshape∆
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2<
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulƒ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2»
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisП
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Є
3transformer_block_7/sequential_7/dense_23/TensordotReshapeDtransformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@25
3transformer_block_7/sequential_7/dense_23/TensordotК
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpѓ
1transformer_block_7/sequential_7/dense_23/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_23/Tensordot:output:0Htransformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@23
1transformer_block_7/sequential_7/dense_23/BiasAddЏ
.transformer_block_7/sequential_7/dense_23/ReluRelu:transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@20
.transformer_block_7/sequential_7/dense_23/ReluФ
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpЊ
8transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_24/Tensordot/axes≈
8transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_24/Tensordot/freeв
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShape<transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/Shape»
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2ћ
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1ј
9transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_24/Tensordot/Const®
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_24/Tensordot/Prodƒ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1∞
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ƒ
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisВ
:transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_24/Tensordot/concatі
9transformer_block_7/sequential_7/dense_24/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/stackƒ
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose	Transpose<transformer_block_7/sequential_7/dense_23/Relu:activations:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2?
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose«
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Reshape∆
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulƒ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2»
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisП
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Є
3transformer_block_7/sequential_7/dense_24/TensordotReshapeDtransformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 25
3transformer_block_7/sequential_7/dense_24/TensordotК
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpѓ
1transformer_block_7/sequential_7/dense_24/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_24/Tensordot:output:0Htransformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 23
1transformer_block_7/sequential_7/dense_24/BiasAdd–
'transformer_block_7/dropout_21/IdentityIdentity:transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2)
'transformer_block_7/dropout_21/Identityз
transformer_block_7/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:00transformer_block_7/dropout_21/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
transformer_block_7/add_1а
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesі
7transformer_block_7/layer_normalization_15/moments/meanMeantransformer_block_7/add_1:z:0Rtransformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(29
7transformer_block_7/layer_normalization_15/moments/meanК
?transformer_block_7/layer_normalization_15/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2A
?transformer_block_7/layer_normalization_15/moments/StopGradientј
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add_1:z:0Htransformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2F
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceи
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesл
;transformer_block_7/layer_normalization_15/moments/varianceMeanHtransformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2=
;transformer_block_7/layer_normalization_15/moments/varianceљ
:transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_7/layer_normalization_15/batchnorm/add/yЊ
8transformer_block_7/layer_normalization_15/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_15/moments/variance:output:0Ctransformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2:
8transformer_block_7/layer_normalization_15/batchnorm/addх
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2<
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtЯ
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp¬
8transformer_block_7/layer_normalization_15/batchnorm/mulMul>transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_15/batchnorm/mulТ
:transformer_block_7/layer_normalization_15/batchnorm/mul_1Multransformer_block_7/add_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_15/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_15/moments/mean:output:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_2У
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpЊ
8transformer_block_7/layer_normalization_15/batchnorm/subSubKtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_15/batchnorm/subµ
:transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_15/batchnorm/add_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
flatten_3/ConstЊ
flatten_3/ReshapeReshape>transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisЊ
concatenate_3/concatConcatV2flatten_3/Reshape:output:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€»2
concatenate_3/concat©
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	»@*
dtype02 
dense_25/MatMul/ReadVariableOp•
dense_25/MatMulMatMulconcatenate_3/concat:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_25/MatMulІ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp•
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_25/ReluЕ
dropout_22/IdentityIdentitydense_25/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_22/Identity®
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp§
dense_26/MatMulMatMuldropout_22/Identity:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/MatMulІ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp•
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/ReluЕ
dropout_23/IdentityIdentitydense_26/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_23/Identity®
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_27/MatMul/ReadVariableOp§
dense_27/MatMulMatMuldropout_23/Identity:output:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_27/MatMulІ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_27/BiasAdd/ReadVariableOp•
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_27/BiasAddД
IdentityIdentitydense_27/BiasAdd:output:0/^batch_normalization_6/batchnorm/ReadVariableOp1^batch_normalization_6/batchnorm/ReadVariableOp_11^batch_normalization_6/batchnorm/ReadVariableOp_23^batch_normalization_6/batchnorm/mul/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp1^batch_normalization_7/batchnorm/ReadVariableOp_11^batch_normalization_7/batchnorm/ReadVariableOp_23^batch_normalization_7/batchnorm/mul/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp<^token_and_position_embedding_3/embedding_6/embedding_lookup<^token_and_position_embedding_3/embedding_7/embedding_lookupD^transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpD^transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpO^transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpY^transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpL^transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpA^transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpA^transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2d
0batch_normalization_6/batchnorm/ReadVariableOp_10batch_normalization_6/batchnorm/ReadVariableOp_12d
0batch_normalization_6/batchnorm/ReadVariableOp_20batch_normalization_6/batchnorm/ReadVariableOp_22h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2d
0batch_normalization_7/batchnorm/ReadVariableOp_10batch_normalization_7/batchnorm/ReadVariableOp_12d
0batch_normalization_7/batchnorm/ReadVariableOp_20batch_normalization_7/batchnorm/ReadVariableOp_22h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2z
;token_and_position_embedding_3/embedding_6/embedding_lookup;token_and_position_embedding_3/embedding_6/embedding_lookup2z
;token_and_position_embedding_3/embedding_7/embedding_lookup;token_and_position_embedding_3/embedding_7/embedding_lookup2К
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2Т
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2К
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2Т
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2і
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpAtransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2Ъ
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpKtransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2Ю
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2Ю
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2Д
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2И
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2Д
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2И
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:S O
)
_output_shapes
:€€€€€€€€€†Ь
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
„к
§&
C__inference_model_3_layer_call_and_return_conditional_losses_411485
inputs_0
inputs_1F
Btoken_and_position_embedding_3_embedding_7_embedding_lookup_411187F
Btoken_and_position_embedding_3_embedding_6_embedding_lookup_4111938
4conv1d_6_conv1d_expanddims_1_readvariableop_resource,
(conv1d_6_biasadd_readvariableop_resource8
4conv1d_7_conv1d_expanddims_1_readvariableop_resource,
(conv1d_7_biasadd_readvariableop_resource0
,batch_normalization_6_assignmovingavg_4112432
.batch_normalization_6_assignmovingavg_1_411249?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource0
,batch_normalization_7_assignmovingavg_4112752
.batch_normalization_7_assignmovingavg_1_411281?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resourceZ
Vtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_query_add_readvariableop_resourceX
Ttransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resourceN
Jtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resourceZ
Vtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resourceP
Ltransformer_block_7_multi_head_attention_7_value_add_readvariableop_resourcee
atransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource[
Wtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resourceO
Ktransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resourceM
Itransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resourceT
Ptransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resourceP
Ltransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource+
'dense_26_matmul_readvariableop_resource,
(dense_26_biasadd_readvariableop_resource+
'dense_27_matmul_readvariableop_resource,
(dense_27_biasadd_readvariableop_resource
identityИҐ9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_6/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_6/batchnorm/ReadVariableOpҐ2batch_normalization_6/batchnorm/mul/ReadVariableOpҐ9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpҐ4batch_normalization_7/AssignMovingAvg/ReadVariableOpҐ;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpҐ6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpҐ.batch_normalization_7/batchnorm/ReadVariableOpҐ2batch_normalization_7/batchnorm/mul/ReadVariableOpҐconv1d_6/BiasAdd/ReadVariableOpҐ+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpҐconv1d_7/BiasAdd/ReadVariableOpҐ+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpҐdense_25/BiasAdd/ReadVariableOpҐdense_25/MatMul/ReadVariableOpҐdense_26/BiasAdd/ReadVariableOpҐdense_26/MatMul/ReadVariableOpҐdense_27/BiasAdd/ReadVariableOpҐdense_27/MatMul/ReadVariableOpҐ;token_and_position_embedding_3/embedding_6/embedding_lookupҐ;token_and_position_embedding_3/embedding_7/embedding_lookupҐCtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpҐGtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpҐCtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpҐGtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpҐNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpҐXtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpҐAtransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpҐKtransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpҐCtransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpҐMtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpҐCtransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpҐMtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpҐ@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpҐBtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpҐ@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpҐBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpД
$token_and_position_embedding_3/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_3/Shapeї
2token_and_position_embedding_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€24
2token_and_position_embedding_3/strided_slice/stackґ
4token_and_position_embedding_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_3/strided_slice/stack_1ґ
4token_and_position_embedding_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_3/strided_slice/stack_2Ь
,token_and_position_embedding_3/strided_sliceStridedSlice-token_and_position_embedding_3/Shape:output:0;token_and_position_embedding_3/strided_slice/stack:output:0=token_and_position_embedding_3/strided_slice/stack_1:output:0=token_and_position_embedding_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_3/strided_sliceЪ
*token_and_position_embedding_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_3/range/startЪ
*token_and_position_embedding_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_3/range/deltaЫ
$token_and_position_embedding_3/rangeRange3token_and_position_embedding_3/range/start:output:05token_and_position_embedding_3/strided_slice:output:03token_and_position_embedding_3/range/delta:output:0*#
_output_shapes
:€€€€€€€€€2&
$token_and_position_embedding_3/range 
;token_and_position_embedding_3/embedding_7/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_7_embedding_lookup_411187-token_and_position_embedding_3/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411187*'
_output_shapes
:€€€€€€€€€ *
dtype02=
;token_and_position_embedding_3/embedding_7/embedding_lookupХ
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_7/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_7/embedding_lookup/411187*'
_output_shapes
:€€€€€€€€€ 2F
Dtoken_and_position_embedding_3/embedding_7/embedding_lookup/IdentityЭ
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2H
Ftoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1Ј
/token_and_position_embedding_3/embedding_6/CastCastinputs_0*

DstT0*

SrcT0*)
_output_shapes
:€€€€€€€€€†Ь21
/token_and_position_embedding_3/embedding_6/Cast÷
;token_and_position_embedding_3/embedding_6/embedding_lookupResourceGatherBtoken_and_position_embedding_3_embedding_6_embedding_lookup_4111933token_and_position_embedding_3/embedding_6/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411193*-
_output_shapes
:€€€€€€€€€†Ь *
dtype02=
;token_and_position_embedding_3/embedding_6/embedding_lookupЫ
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_3/embedding_6/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_3/embedding_6/embedding_lookup/411193*-
_output_shapes
:€€€€€€€€€†Ь 2F
Dtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity£
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2H
Ftoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1Ђ
"token_and_position_embedding_3/addAddV2Otoken_and_position_embedding_3/embedding_6/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_3/embedding_7/embedding_lookup/Identity_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2$
"token_and_position_embedding_3/addЛ
conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_6/conv1d/ExpandDims/dim”
conv1d_6/conv1d/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0'conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2
conv1d_6/conv1d/ExpandDims”
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype02-
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_6/conv1d/ExpandDims_1/dimџ
conv1d_6/conv1d/ExpandDims_1
ExpandDims3conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  2
conv1d_6/conv1d/ExpandDims_1№
conv1d_6/conv1dConv2D#conv1d_6/conv1d/ExpandDims:output:0%conv1d_6/conv1d/ExpandDims_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь *
paddingSAME*
strides
2
conv1d_6/conv1dѓ
conv1d_6/conv1d/SqueezeSqueezeconv1d_6/conv1d:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь *
squeeze_dims

э€€€€€€€€2
conv1d_6/conv1d/SqueezeІ
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_6/BiasAdd/ReadVariableOp≤
conv1d_6/BiasAddBiasAdd conv1d_6/conv1d/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
conv1d_6/BiasAddy
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2
conv1d_6/ReluК
"average_pooling1d_9/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_9/ExpandDims/dim‘
average_pooling1d_9/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_9/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2 
average_pooling1d_9/ExpandDimsе
average_pooling1d_9/AvgPoolAvgPool'average_pooling1d_9/ExpandDims:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
ksize
*
paddingVALID*
strides
2
average_pooling1d_9/AvgPoolє
average_pooling1d_9/SqueezeSqueeze$average_pooling1d_9/AvgPool:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims
2
average_pooling1d_9/SqueezeЛ
conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_7/conv1d/ExpandDims/dim–
conv1d_7/conv1d/ExpandDims
ExpandDims$average_pooling1d_9/Squeeze:output:0'conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2
conv1d_7/conv1d/ExpandDims”
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	  *
dtype02-
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_7/conv1d/ExpandDims_1/dimџ
conv1d_7/conv1d/ExpandDims_1
ExpandDims3conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	  2
conv1d_7/conv1d/ExpandDims_1џ
conv1d_7/conv1dConv2D#conv1d_7/conv1d/ExpandDims:output:0%conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ *
paddingSAME*
strides
2
conv1d_7/conv1dЃ
conv1d_7/conv1d/SqueezeSqueezeconv1d_7/conv1d:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ *
squeeze_dims

э€€€€€€€€2
conv1d_7/conv1d/SqueezeІ
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_7/BiasAdd/ReadVariableOp±
conv1d_7/BiasAddBiasAdd conv1d_7/conv1d/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
conv1d_7/BiasAddx
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ъ 2
conv1d_7/ReluМ
#average_pooling1d_11/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_11/ExpandDims/dimв
average_pooling1d_11/ExpandDims
ExpandDims&token_and_position_embedding_3/add:z:0,average_pooling1d_11/ExpandDims/dim:output:0*
T0*1
_output_shapes
:€€€€€€€€€†Ь 2!
average_pooling1d_11/ExpandDimsй
average_pooling1d_11/AvgPoolAvgPool(average_pooling1d_11/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€B *
ksize	
ђ*
paddingVALID*
strides	
ђ2
average_pooling1d_11/AvgPoolї
average_pooling1d_11/SqueezeSqueeze%average_pooling1d_11/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
squeeze_dims
2
average_pooling1d_11/SqueezeМ
#average_pooling1d_10/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#average_pooling1d_10/ExpandDims/dim÷
average_pooling1d_10/ExpandDims
ExpandDimsconv1d_7/Relu:activations:0,average_pooling1d_10/ExpandDims/dim:output:0*
T0*0
_output_shapes
:€€€€€€€€€Ъ 2!
average_pooling1d_10/ExpandDimsз
average_pooling1d_10/AvgPoolAvgPool(average_pooling1d_10/ExpandDims:output:0*
T0*/
_output_shapes
:€€€€€€€€€B *
ksize

*
paddingVALID*
strides

2
average_pooling1d_10/AvgPoolї
average_pooling1d_10/SqueezeSqueeze%average_pooling1d_10/AvgPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
squeeze_dims
2
average_pooling1d_10/Squeezeљ
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_6/moments/mean/reduction_indicesф
"batch_normalization_6/moments/meanMean%average_pooling1d_10/Squeeze:output:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_6/moments/mean¬
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_6/moments/StopGradientЙ
/batch_normalization_6/moments/SquaredDifferenceSquaredDifference%average_pooling1d_10/Squeeze:output:03batch_normalization_6/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 21
/batch_normalization_6/moments/SquaredDifference≈
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_6/moments/variance/reduction_indicesО
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_6/moments/variance√
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_6/moments/SqueezeЋ
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1О
+batch_normalization_6/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_6/AssignMovingAvg/decay’
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_6_assignmovingavg_411243*
_output_shapes
: *
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOpя
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/sub÷
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/mul≥
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_6_assignmovingavg_411243-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_6/AssignMovingAvg/411243*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpФ
-batch_normalization_6/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_6/AssignMovingAvg_1/decayџ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_1_411249*
_output_shapes
: *
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/subа
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/mulњ
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_1_411249/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg_1/411249*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yЏ
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/add•
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/Rsqrtа
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/mulџ
%batch_normalization_6/batchnorm/mul_1Mul%average_pooling1d_10/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_6/batchnorm/mul_1”
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_6/batchnorm/mul_2‘
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpў
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_6/batchnorm/subб
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_6/batchnorm/add_1љ
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_7/moments/mean/reduction_indicesф
"batch_normalization_7/moments/meanMean%average_pooling1d_11/Squeeze:output:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_7/moments/mean¬
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_7/moments/StopGradientЙ
/batch_normalization_7/moments/SquaredDifferenceSquaredDifference%average_pooling1d_11/Squeeze:output:03batch_normalization_7/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 21
/batch_normalization_7/moments/SquaredDifference≈
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_7/moments/variance/reduction_indicesО
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_7/moments/variance√
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_7/moments/SqueezeЋ
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1О
+batch_normalization_7/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_7/AssignMovingAvg/decay’
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_7_assignmovingavg_411275*
_output_shapes
: *
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOpя
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/sub÷
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/mul≥
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_7_assignmovingavg_411275-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_7/AssignMovingAvg/411275*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpФ
-batch_normalization_7/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_7/AssignMovingAvg_1/decayџ
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_1_411281*
_output_shapes
: *
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpй
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/subа
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/mulњ
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_1_411281/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg_1/411281*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/yЏ
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/add•
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/Rsqrtа
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOpЁ
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/mulџ
%batch_normalization_7/batchnorm/mul_1Mul%average_pooling1d_11/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_7/batchnorm/mul_1”
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_7/batchnorm/mul_2‘
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpў
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_7/batchnorm/subб
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2'
%batch_normalization_7/batchnorm/add_1Ђ
	add_3/addAddV2)batch_normalization_6/batchnorm/add_1:z:0)batch_normalization_7/batchnorm/add_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
	add_3/addє
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp–
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/query/einsum/EinsumЧ
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_query_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp≈
4transformer_block_7/multi_head_attention_7/query/addAddV2Gtransformer_block_7/multi_head_attention_7/query/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 26
4transformer_block_7/multi_head_attention_7/query/add≥
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpReadVariableOpTtransformer_block_7_multi_head_attention_7_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02M
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp 
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumEinsumadd_3/add:z:0Stransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2>
<transformer_block_7/multi_head_attention_7/key/einsum/EinsumС
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpReadVariableOpJtransformer_block_7_multi_head_attention_7_key_add_readvariableop_resource*
_output_shapes

: *
dtype02C
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpљ
2transformer_block_7/multi_head_attention_7/key/addAddV2Etransformer_block_7/multi_head_attention_7/key/einsum/Einsum:output:0Itransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 24
2transformer_block_7/multi_head_attention_7/key/addє
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpReadVariableOpVtransformer_block_7_multi_head_attention_7_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02O
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp–
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumEinsumadd_3/add:z:0Utransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationabc,cde->abde2@
>transformer_block_7/multi_head_attention_7/value/einsum/EinsumЧ
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpReadVariableOpLtransformer_block_7_multi_head_attention_7_value_add_readvariableop_resource*
_output_shapes

: *
dtype02E
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp≈
4transformer_block_7/multi_head_attention_7/value/addAddV2Gtransformer_block_7/multi_head_attention_7/value/einsum/Einsum:output:0Ktransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€B 26
4transformer_block_7/multi_head_attention_7/value/add©
0transformer_block_7/multi_head_attention_7/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *у5>22
0transformer_block_7/multi_head_attention_7/Mul/yЦ
.transformer_block_7/multi_head_attention_7/MulMul8transformer_block_7/multi_head_attention_7/query/add:z:09transformer_block_7/multi_head_attention_7/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€B 20
.transformer_block_7/multi_head_attention_7/Mulћ
8transformer_block_7/multi_head_attention_7/einsum/EinsumEinsum6transformer_block_7/multi_head_attention_7/key/add:z:02transformer_block_7/multi_head_attention_7/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€BB*
equationaecd,abcd->acbe2:
8transformer_block_7/multi_head_attention_7/einsum/EinsumА
:transformer_block_7/multi_head_attention_7/softmax/SoftmaxSoftmaxAtransformer_block_7/multi_head_attention_7/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2<
:transformer_block_7/multi_head_attention_7/softmax/Softmax…
@transformer_block_7/multi_head_attention_7/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Const“
>transformer_block_7/multi_head_attention_7/dropout/dropout/MulMulDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0Itransformer_block_7/multi_head_attention_7/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2@
>transformer_block_7/multi_head_attention_7/dropout/dropout/Mulш
@transformer_block_7/multi_head_attention_7/dropout/dropout/ShapeShapeDtransformer_block_7/multi_head_attention_7/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Shape’
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformRandomUniformItransformer_block_7/multi_head_attention_7/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB*
dtype02Y
Wtransformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniformџ
Itransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Itransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual/yТ
Gtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqualGreaterEqual`transformer_block_7/multi_head_attention_7/dropout/dropout/random_uniform/RandomUniform:output:0Rtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€BB2I
Gtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual†
?transformer_block_7/multi_head_attention_7/dropout/dropout/CastCastKtransformer_block_7/multi_head_attention_7/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€BB2A
?transformer_block_7/multi_head_attention_7/dropout/dropout/Castќ
@transformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1MulBtransformer_block_7/multi_head_attention_7/dropout/dropout/Mul:z:0Ctransformer_block_7/multi_head_attention_7/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€BB2B
@transformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1д
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumEinsumDtransformer_block_7/multi_head_attention_7/dropout/dropout/Mul_1:z:08transformer_block_7/multi_head_attention_7/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€B *
equationacbe,aecd->abcd2<
:transformer_block_7/multi_head_attention_7/einsum_1/EinsumЏ
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpatransformer_block_7_multi_head_attention_7_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:  *
dtype02Z
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp£
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/EinsumEinsumCtransformer_block_7/multi_head_attention_7/einsum_1/Einsum:output:0`transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€B *
equationabcd,cde->abe2K
Itransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsumі
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpReadVariableOpWtransformer_block_7_multi_head_attention_7_attention_output_add_readvariableop_resource*
_output_shapes
: *
dtype02P
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpн
?transformer_block_7/multi_head_attention_7/attention_output/addAddV2Rtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum:output:0Vtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2A
?transformer_block_7/multi_head_attention_7/attention_output/add°
,transformer_block_7/dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2.
,transformer_block_7/dropout_20/dropout/ConstС
*transformer_block_7/dropout_20/dropout/MulMulCtransformer_block_7/multi_head_attention_7/attention_output/add:z:05transformer_block_7/dropout_20/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2,
*transformer_block_7/dropout_20/dropout/Mulѕ
,transformer_block_7/dropout_20/dropout/ShapeShapeCtransformer_block_7/multi_head_attention_7/attention_output/add:z:0*
T0*
_output_shapes
:2.
,transformer_block_7/dropout_20/dropout/ShapeХ
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_20/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
dtype02E
Ctransformer_block_7/dropout_20/dropout/random_uniform/RandomUniform≥
5transformer_block_7/dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=27
5transformer_block_7/dropout_20/dropout/GreaterEqual/yЊ
3transformer_block_7/dropout_20/dropout/GreaterEqualGreaterEqualLtransformer_block_7/dropout_20/dropout/random_uniform/RandomUniform:output:0>transformer_block_7/dropout_20/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 25
3transformer_block_7/dropout_20/dropout/GreaterEqualа
+transformer_block_7/dropout_20/dropout/CastCast7transformer_block_7/dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€B 2-
+transformer_block_7/dropout_20/dropout/Castъ
,transformer_block_7/dropout_20/dropout/Mul_1Mul.transformer_block_7/dropout_20/dropout/Mul:z:0/transformer_block_7/dropout_20/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€B 2.
,transformer_block_7/dropout_20/dropout/Mul_1≤
transformer_block_7/addAddV2add_3/add:z:00transformer_block_7/dropout_20/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
transformer_block_7/addа
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_14/moments/mean/reduction_indices≤
7transformer_block_7/layer_normalization_14/moments/meanMeantransformer_block_7/add:z:0Rtransformer_block_7/layer_normalization_14/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(29
7transformer_block_7/layer_normalization_14/moments/meanК
?transformer_block_7/layer_normalization_14/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_14/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2A
?transformer_block_7/layer_normalization_14/moments/StopGradientЊ
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add:z:0Htransformer_block_7/layer_normalization_14/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2F
Dtransformer_block_7/layer_normalization_14/moments/SquaredDifferenceи
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_14/moments/variance/reduction_indicesл
;transformer_block_7/layer_normalization_14/moments/varianceMeanHtransformer_block_7/layer_normalization_14/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_14/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2=
;transformer_block_7/layer_normalization_14/moments/varianceљ
:transformer_block_7/layer_normalization_14/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_7/layer_normalization_14/batchnorm/add/yЊ
8transformer_block_7/layer_normalization_14/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_14/moments/variance:output:0Ctransformer_block_7/layer_normalization_14/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2:
8transformer_block_7/layer_normalization_14/batchnorm/addх
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_14/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2<
:transformer_block_7/layer_normalization_14/batchnorm/RsqrtЯ
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_14_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp¬
8transformer_block_7/layer_normalization_14/batchnorm/mulMul>transformer_block_7/layer_normalization_14/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_14/batchnorm/mulР
:transformer_block_7/layer_normalization_14/batchnorm/mul_1Multransformer_block_7/add:z:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_14/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_14/moments/mean:output:0<transformer_block_7/layer_normalization_14/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_14/batchnorm/mul_2У
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_14_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpЊ
8transformer_block_7/layer_normalization_14/batchnorm/subSubKtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_14/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_14/batchnorm/subµ
:transformer_block_7/layer_normalization_14/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_14/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_14/batchnorm/add_1Ф
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02D
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpЊ
8transformer_block_7/sequential_7/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_23/Tensordot/axes≈
8transformer_block_7/sequential_7/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_23/Tensordot/freeд
9transformer_block_7/sequential_7/dense_23/Tensordot/ShapeShape>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/Shape»
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2ћ
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_23/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1ј
9transformer_block_7/sequential_7/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_23/Tensordot/Const®
8transformer_block_7/sequential_7/dense_23/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_23/Tensordot/Prodƒ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_1∞
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_23/Tensordot/Prod_1ƒ
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_23/Tensordot/concat/axisВ
:transformer_block_7/sequential_7/dense_23/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_23/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_23/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_23/Tensordot/concatі
9transformer_block_7/sequential_7/dense_23/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_23/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_23/Tensordot/stack∆
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose	Transpose>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:0Ctransformer_block_7/sequential_7/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2?
=transformer_block_7/sequential_7/dense_23/Tensordot/transpose«
;transformer_block_7/sequential_7/dense_23/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_23/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Reshape∆
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_23/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2<
:transformer_block_7/sequential_7/dense_23/Tensordot/MatMulƒ
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;transformer_block_7/sequential_7/dense_23/Tensordot/Const_2»
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axisП
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_23/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_23/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_23/Tensordot/concat_1Є
3transformer_block_7/sequential_7/dense_23/TensordotReshapeDtransformer_block_7/sequential_7/dense_23/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@25
3transformer_block_7/sequential_7/dense_23/TensordotК
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpѓ
1transformer_block_7/sequential_7/dense_23/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_23/Tensordot:output:0Htransformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@23
1transformer_block_7/sequential_7/dense_23/BiasAddЏ
.transformer_block_7/sequential_7/dense_23/ReluRelu:transformer_block_7/sequential_7/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@20
.transformer_block_7/sequential_7/dense_23/ReluФ
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_7_sequential_7_dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02D
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpЊ
8transformer_block_7/sequential_7/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_7/sequential_7/dense_24/Tensordot/axes≈
8transformer_block_7/sequential_7/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_7/sequential_7/dense_24/Tensordot/freeв
9transformer_block_7/sequential_7/dense_24/Tensordot/ShapeShape<transformer_block_7/sequential_7/dense_23/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/Shape»
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis£
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2ћ
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis©
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1GatherV2Btransformer_block_7/sequential_7/dense_24/Tensordot/Shape:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Ltransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1ј
9transformer_block_7/sequential_7/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_7/sequential_7/dense_24/Tensordot/Const®
8transformer_block_7/sequential_7/dense_24/Tensordot/ProdProdEtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Btransformer_block_7/sequential_7/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_7/sequential_7/dense_24/Tensordot/Prodƒ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_1∞
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ProdGtransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2_1:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/Prod_1ƒ
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_7/sequential_7/dense_24/Tensordot/concat/axisВ
:transformer_block_7/sequential_7/dense_24/Tensordot/concatConcatV2Atransformer_block_7/sequential_7/dense_24/Tensordot/free:output:0Atransformer_block_7/sequential_7/dense_24/Tensordot/axes:output:0Htransformer_block_7/sequential_7/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_7/sequential_7/dense_24/Tensordot/concatі
9transformer_block_7/sequential_7/dense_24/Tensordot/stackPackAtransformer_block_7/sequential_7/dense_24/Tensordot/Prod:output:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_7/sequential_7/dense_24/Tensordot/stackƒ
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose	Transpose<transformer_block_7/sequential_7/dense_23/Relu:activations:0Ctransformer_block_7/sequential_7/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2?
=transformer_block_7/sequential_7/dense_24/Tensordot/transpose«
;transformer_block_7/sequential_7/dense_24/Tensordot/ReshapeReshapeAtransformer_block_7/sequential_7/dense_24/Tensordot/transpose:y:0Btransformer_block_7/sequential_7/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Reshape∆
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulMatMulDtransformer_block_7/sequential_7/dense_24/Tensordot/Reshape:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2<
:transformer_block_7/sequential_7/dense_24/Tensordot/MatMulƒ
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_7/sequential_7/dense_24/Tensordot/Const_2»
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axisП
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1ConcatV2Etransformer_block_7/sequential_7/dense_24/Tensordot/GatherV2:output:0Dtransformer_block_7/sequential_7/dense_24/Tensordot/Const_2:output:0Jtransformer_block_7/sequential_7/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_7/sequential_7/dense_24/Tensordot/concat_1Є
3transformer_block_7/sequential_7/dense_24/TensordotReshapeDtransformer_block_7/sequential_7/dense_24/Tensordot/MatMul:product:0Etransformer_block_7/sequential_7/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 25
3transformer_block_7/sequential_7/dense_24/TensordotК
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_7_sequential_7_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpѓ
1transformer_block_7/sequential_7/dense_24/BiasAddBiasAdd<transformer_block_7/sequential_7/dense_24/Tensordot:output:0Htransformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 23
1transformer_block_7/sequential_7/dense_24/BiasAdd°
,transformer_block_7/dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2.
,transformer_block_7/dropout_21/dropout/ConstИ
*transformer_block_7/dropout_21/dropout/MulMul:transformer_block_7/sequential_7/dense_24/BiasAdd:output:05transformer_block_7/dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2,
*transformer_block_7/dropout_21/dropout/Mul∆
,transformer_block_7/dropout_21/dropout/ShapeShape:transformer_block_7/sequential_7/dense_24/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_7/dropout_21/dropout/ShapeХ
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_7/dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€B *
dtype02E
Ctransformer_block_7/dropout_21/dropout/random_uniform/RandomUniform≥
5transformer_block_7/dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=27
5transformer_block_7/dropout_21/dropout/GreaterEqual/yЊ
3transformer_block_7/dropout_21/dropout/GreaterEqualGreaterEqualLtransformer_block_7/dropout_21/dropout/random_uniform/RandomUniform:output:0>transformer_block_7/dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 25
3transformer_block_7/dropout_21/dropout/GreaterEqualа
+transformer_block_7/dropout_21/dropout/CastCast7transformer_block_7/dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:€€€€€€€€€B 2-
+transformer_block_7/dropout_21/dropout/Castъ
,transformer_block_7/dropout_21/dropout/Mul_1Mul.transformer_block_7/dropout_21/dropout/Mul:z:0/transformer_block_7/dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:€€€€€€€€€B 2.
,transformer_block_7/dropout_21/dropout/Mul_1з
transformer_block_7/add_1AddV2>transformer_block_7/layer_normalization_14/batchnorm/add_1:z:00transformer_block_7/dropout_21/dropout/Mul_1:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
transformer_block_7/add_1а
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_7/layer_normalization_15/moments/mean/reduction_indicesі
7transformer_block_7/layer_normalization_15/moments/meanMeantransformer_block_7/add_1:z:0Rtransformer_block_7/layer_normalization_15/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(29
7transformer_block_7/layer_normalization_15/moments/meanК
?transformer_block_7/layer_normalization_15/moments/StopGradientStopGradient@transformer_block_7/layer_normalization_15/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2A
?transformer_block_7/layer_normalization_15/moments/StopGradientј
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceSquaredDifferencetransformer_block_7/add_1:z:0Htransformer_block_7/layer_normalization_15/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2F
Dtransformer_block_7/layer_normalization_15/moments/SquaredDifferenceи
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_7/layer_normalization_15/moments/variance/reduction_indicesл
;transformer_block_7/layer_normalization_15/moments/varianceMeanHtransformer_block_7/layer_normalization_15/moments/SquaredDifference:z:0Vtransformer_block_7/layer_normalization_15/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€B*
	keep_dims(2=
;transformer_block_7/layer_normalization_15/moments/varianceљ
:transformer_block_7/layer_normalization_15/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52<
:transformer_block_7/layer_normalization_15/batchnorm/add/yЊ
8transformer_block_7/layer_normalization_15/batchnorm/addAddV2Dtransformer_block_7/layer_normalization_15/moments/variance:output:0Ctransformer_block_7/layer_normalization_15/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€B2:
8transformer_block_7/layer_normalization_15/batchnorm/addх
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtRsqrt<transformer_block_7/layer_normalization_15/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€B2<
:transformer_block_7/layer_normalization_15/batchnorm/RsqrtЯ
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_7_layer_normalization_15_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp¬
8transformer_block_7/layer_normalization_15/batchnorm/mulMul>transformer_block_7/layer_normalization_15/batchnorm/Rsqrt:y:0Otransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_15/batchnorm/mulТ
:transformer_block_7/layer_normalization_15/batchnorm/mul_1Multransformer_block_7/add_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_1µ
:transformer_block_7/layer_normalization_15/batchnorm/mul_2Mul@transformer_block_7/layer_normalization_15/moments/mean:output:0<transformer_block_7/layer_normalization_15/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_15/batchnorm/mul_2У
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_7_layer_normalization_15_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpЊ
8transformer_block_7/layer_normalization_15/batchnorm/subSubKtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp:value:0>transformer_block_7/layer_normalization_15/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2:
8transformer_block_7/layer_normalization_15/batchnorm/subµ
:transformer_block_7/layer_normalization_15/batchnorm/add_1AddV2>transformer_block_7/layer_normalization_15/batchnorm/mul_1:z:0<transformer_block_7/layer_normalization_15/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2<
:transformer_block_7/layer_normalization_15/batchnorm/add_1s
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
flatten_3/ConstЊ
flatten_3/ReshapeReshape>transformer_block_7/layer_normalization_15/batchnorm/add_1:z:0flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј2
flatten_3/Reshapex
concatenate_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_3/concat/axisЊ
concatenate_3/concatConcatV2flatten_3/Reshape:output:0inputs_1"concatenate_3/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€»2
concatenate_3/concat©
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	»@*
dtype02 
dense_25/MatMul/ReadVariableOp•
dense_25/MatMulMatMulconcatenate_3/concat:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_25/MatMulІ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp•
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_25/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *эJБ?2
dropout_22/dropout/Const©
dropout_22/dropout/MulMuldense_25/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape’
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype021
/dropout_22/dropout/random_uniform/RandomUniformЛ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dropout_22/dropout/GreaterEqual/yк
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
dropout_22/dropout/GreaterEqual†
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_22/dropout/Cast¶
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_22/dropout/Mul_1®
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp§
dense_26/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/MatMulІ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp•
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_26/Reluy
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *эJБ?2
dropout_23/dropout/Const©
dropout_23/dropout/MulMuldense_26/Relu:activations:0!dropout_23/dropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_23/dropout/Mul
dropout_23/dropout/ShapeShapedense_26/Relu:activations:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shape’
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype021
/dropout_23/dropout/random_uniform/RandomUniformЛ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2#
!dropout_23/dropout/GreaterEqual/yк
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
dropout_23/dropout/GreaterEqual†
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_23/dropout/Cast¶
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_23/dropout/Mul_1®
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_27/MatMul/ReadVariableOp§
dense_27/MatMulMatMuldropout_23/dropout/Mul_1:z:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_27/MatMulІ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_27/BiasAdd/ReadVariableOp•
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_27/BiasAddМ
IdentityIdentitydense_27/BiasAdd:output:0:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_6/batchnorm/ReadVariableOp3^batch_normalization_6/batchnorm/mul/ReadVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_7/batchnorm/ReadVariableOp3^batch_normalization_7/batchnorm/mul/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/conv1d/ExpandDims_1/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp<^token_and_position_embedding_3/embedding_6/embedding_lookup<^token_and_position_embedding_3/embedding_7/embedding_lookupD^transformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpD^transformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpH^transformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpO^transformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpY^transformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpB^transformer_block_7/multi_head_attention_7/key/add/ReadVariableOpL^transformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/query/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpD^transformer_block_7/multi_head_attention_7/value/add/ReadVariableOpN^transformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpA^transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpA^transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOpC^transformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_6/batchnorm/ReadVariableOp.batch_normalization_6/batchnorm/ReadVariableOp2h
2batch_normalization_6/batchnorm/mul/ReadVariableOp2batch_normalization_6/batchnorm/mul/ReadVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_7/batchnorm/ReadVariableOp.batch_normalization_7/batchnorm/ReadVariableOp2h
2batch_normalization_7/batchnorm/mul/ReadVariableOp2batch_normalization_7/batchnorm/mul/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp+conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp+conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2z
;token_and_position_embedding_3/embedding_6/embedding_lookup;token_and_position_embedding_3/embedding_6/embedding_lookup2z
;token_and_position_embedding_3/embedding_7/embedding_lookup;token_and_position_embedding_3/embedding_7/embedding_lookup2К
Ctransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_14/batchnorm/ReadVariableOp2Т
Gtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_14/batchnorm/mul/ReadVariableOp2К
Ctransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOpCtransformer_block_7/layer_normalization_15/batchnorm/ReadVariableOp2Т
Gtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOpGtransformer_block_7/layer_normalization_15/batchnorm/mul/ReadVariableOp2†
Ntransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOpNtransformer_block_7/multi_head_attention_7/attention_output/add/ReadVariableOp2і
Xtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOpXtransformer_block_7/multi_head_attention_7/attention_output/einsum/Einsum/ReadVariableOp2Ж
Atransformer_block_7/multi_head_attention_7/key/add/ReadVariableOpAtransformer_block_7/multi_head_attention_7/key/add/ReadVariableOp2Ъ
Ktransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOpKtransformer_block_7/multi_head_attention_7/key/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_7/multi_head_attention_7/query/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/query/add/ReadVariableOp2Ю
Mtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/query/einsum/Einsum/ReadVariableOp2К
Ctransformer_block_7/multi_head_attention_7/value/add/ReadVariableOpCtransformer_block_7/multi_head_attention_7/value/add/ReadVariableOp2Ю
Mtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOpMtransformer_block_7/multi_head_attention_7/value/einsum/Einsum/ReadVariableOp2Д
@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_23/BiasAdd/ReadVariableOp2И
Btransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_23/Tensordot/ReadVariableOp2Д
@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp@transformer_block_7/sequential_7/dense_24/BiasAdd/ReadVariableOp2И
Btransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOpBtransformer_block_7/sequential_7/dense_24/Tensordot/ReadVariableOp:S O
)
_output_shapes
:€€€€€€€€€†Ь
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs/1
…
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_412717

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
В
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_412712

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *эJБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
с	
Ё
D__inference_dense_25_layer_call_and_return_conditional_losses_410520

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	»@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€»::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€»
 
_user_specified_nameinputs
Б
Й
H__inference_sequential_7_layer_call_and_return_conditional_losses_409731
dense_23_input
dense_23_409720
dense_23_409722
dense_24_409725
dense_24_409727
identityИҐ dense_23/StatefulPartitionedCallҐ dense_24/StatefulPartitionedCall£
 dense_23/StatefulPartitionedCallStatefulPartitionedCalldense_23_inputdense_23_409720dense_23_409722*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_4096542"
 dense_23/StatefulPartitionedCallЊ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_409725dense_24_409727*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_4097002"
 dense_24/StatefulPartitionedCall«
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€B 
(
_user_specified_namedense_23_input
В
e
F__inference_dropout_23_layer_call_and_return_conditional_losses_410605

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *эJБ?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeі
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
dropout/GreaterEqual/yЊ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
•
d
+__inference_dropout_22_layer_call_fn_412722

inputs
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_4105482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
НJ
ѓ
H__inference_sequential_7_layer_call_and_return_conditional_losses_412907

inputs.
*dense_23_tensordot_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identityИҐdense_23/BiasAdd/ReadVariableOpҐ!dense_23/Tensordot/ReadVariableOpҐdense_24/BiasAdd/ReadVariableOpҐ!dense_24/Tensordot/ReadVariableOp±
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axesГ
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/freej
dense_23/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_23/Tensordot/ShapeЖ
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axisю
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2К
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axisД
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const§
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/ProdВ
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1ђ
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1В
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axisЁ
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat∞
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stackЂ
dense_23/Tensordot/transpose	Transposeinputs"dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dense_23/Tensordot/transpose√
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_23/Tensordot/Reshape¬
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_23/Tensordot/MatMulВ
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_23/Tensordot/Const_2Ж
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axisк
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1і
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_23/TensordotІ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOpЂ
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_23/BiasAddw
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_23/Relu±
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axesГ
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/free
dense_24/Tensordot/ShapeShapedense_23/Relu:activations:0*
T0*
_output_shapes
:2
dense_24/Tensordot/ShapeЖ
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axisю
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2К
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axisД
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2_1~
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const§
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/ProdВ
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1ђ
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1В
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axisЁ
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat∞
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stackј
dense_24/Tensordot/transpose	Transposedense_23/Relu:activations:0"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€B@2
dense_24/Tensordot/transpose√
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_24/Tensordot/Reshape¬
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
dense_24/Tensordot/MatMulВ
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_2Ж
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axisк
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1і
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dense_24/TensordotІ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOpЂ
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
dense_24/BiasAddэ
IdentityIdentitydense_24/BiasAdd:output:0 ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp"^dense_24/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2F
!dense_24/Tensordot/ReadVariableOp!dense_24/Tensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
∞
Ю
$__inference_signature_wrapper_411175
input_7
input_8
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
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*F
_read_only_resource_inputs(
&$	
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_4092942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:€€€€€€€€€†Ь
!
_user_specified_name	input_7:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_8
Ї
s
I__inference_concatenate_3_layer_call_and_return_conditional_losses_410500

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisА
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€»2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€»2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€ј:€€€€€€€€€:P L
(
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
я
~
)__inference_dense_27_layer_call_fn_412793

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_4106332
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
“
Ґ
(__inference_model_3_layer_call_fn_410917
input_7
input_8
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
identityИҐStatefulPartitionedCall–
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*B
_read_only_resource_inputs$
" 
 !"#$%*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_4108422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*Ќ
_input_shapesї
Є:€€€€€€€€€†Ь:€€€€€€€€€::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:€€€€€€€€€†Ь
!
_user_specified_name	input_7:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_8
Н
П
?__inference_token_and_position_embedding_3_layer_call_fn_411917
x
unknown
	unknown_0
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€†Ь *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *c
f^R\
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_4098152
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€†Ь 2

Identity"
identityIdentity:output:0*0
_input_shapes
:€€€€€€€€€†Ь::22
StatefulPartitionedCallStatefulPartitionedCall:L H
)
_output_shapes
:€€€€€€€€€†Ь

_user_specified_namex
–

а
4__inference_transformer_block_7_layer_call_fn_412656

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
identityИҐStatefulPartitionedCallЅ
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
:€€€€€€€€€B *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_4103702
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:€€€€€€€€€B ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs
–
®
-__inference_sequential_7_layer_call_fn_409786
dense_23_input
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCalldense_23_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€B *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_sequential_7_layer_call_and_return_conditional_losses_4097752
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:€€€€€€€€€B 
(
_user_specified_namedense_23_input
м
©
6__inference_batch_normalization_7_layer_call_fn_412200

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4095752
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Љ0
»
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412249

inputs
assignmovingavg_412224
assignmovingavg_1_412230)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐAssignMovingAvg/ReadVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpҐ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indicesґ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1ћ
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_412224*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpс
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
: 2
AssignMovingAvg/subи
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
: 2
AssignMovingAvg/mulѓ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_412224AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*)
_class
loc:@AssignMovingAvg/412224*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp“
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_412230*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpы
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
: 2
AssignMovingAvg_1/subт
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
: 2
AssignMovingAvg_1/mulї
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_412230AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*+
_class!
loc:@AssignMovingAvg_1/412230*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€B 2
batchnorm/add_1Ј
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€B 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':€€€€€€€€€B ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€B 
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*к
serving_default÷
=
input_72
serving_default_input_7:0€€€€€€€€€†Ь
;
input_80
serving_default_input_8:0€€€€€€€€€<
dense_270
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:иь
‘G
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
±__call__
+≤&call_and_return_all_conditional_losses
≥_default_save_signature"ЬB
_tf_keras_networkАB{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_3", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_6", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_9", "inbound_nodes": [[["conv1d_6", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_7", "inbound_nodes": [[["average_pooling1d_9", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_10", "inbound_nodes": [[["conv1d_7", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_11", "inbound_nodes": [[["token_and_position_embedding_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["average_pooling1d_10", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["average_pooling1d_11", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}], ["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_7", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["transformer_block_7", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_3", "inbound_nodes": [[["flatten_3", 0, 0, {}], ["input_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_25", "inbound_nodes": [[["concatenate_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_22", "inbound_nodes": [[["dense_25", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_26", "inbound_nodes": [[["dropout_22", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}, "name": "dropout_23", "inbound_nodes": [[["dense_26", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_27", "inbound_nodes": [[["dropout_23", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["dense_27", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20000]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 8]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 20000]}, {"class_name": "TensorShape", "items": [null, 8]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.0010000000474974513, "decay": 0.0, "momentum": 0.8999999761581421, "nesterov": false}}}}
с"о
_tf_keras_input_layerќ{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
з
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
і__call__
+µ&call_and_return_all_conditional_losses"Ї
_tf_keras_layer†{"class_name": "TokenAndPositionEmbedding", "name": "token_and_position_embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
й	

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses"¬
_tf_keras_layer®{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20000, 32]}}
Й
&trainable_variables
'regularization_losses
(	variables
)	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"ш
_tf_keras_layerё{"class_name": "AveragePooling1D", "name": "average_pooling1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_9", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [30]}, "pool_size": {"class_name": "__tuple__", "items": [30]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
з	

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"ј
_tf_keras_layer¶{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [9]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 666, 32]}}
Л
0trainable_variables
1regularization_losses
2	variables
3	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"ъ
_tf_keras_layerа{"class_name": "AveragePooling1D", "name": "average_pooling1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_10", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [10]}, "pool_size": {"class_name": "__tuple__", "items": [10]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Н
4trainable_variables
5regularization_losses
6	variables
7	keras_api
Њ__call__
+њ&call_and_return_all_conditional_losses"ь
_tf_keras_layerв{"class_name": "AveragePooling1D", "name": "average_pooling1d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_11", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [300]}, "pool_size": {"class_name": "__tuple__", "items": [300]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Є	
8axis
	9gamma
:beta
;moving_mean
<moving_variance
=trainable_variables
>regularization_losses
?	variables
@	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
Є	
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
¬__call__
+√&call_and_return_all_conditional_losses"в
_tf_keras_layer»{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
≥
Jtrainable_variables
Kregularization_losses
L	variables
M	keras_api
ƒ__call__
+≈&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 66, 32]}, {"class_name": "TensorShape", "items": [null, 66, 32]}]}
Д
Natt
Offn
P
layernorm1
Q
layernorm2
Rdropout1
Sdropout2
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
∆__call__
+«&call_and_return_all_conditional_losses"•
_tf_keras_layerЛ{"class_name": "TransformerBlock", "name": "transformer_block_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
и
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
»__call__
+…&call_and_return_all_conditional_losses"„
_tf_keras_layerљ{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
й"ж
_tf_keras_input_layer∆{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 8]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}
–
\trainable_variables
]regularization_losses
^	variables
_	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses"њ
_tf_keras_layer•{"class_name": "Concatenate", "name": "concatenate_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_3", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2112]}, {"class_name": "TensorShape", "items": [null, 8]}]}
ш

`kernel
abias
btrainable_variables
cregularization_losses
d	variables
e	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2120]}}
к
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses"ў
_tf_keras_layerњ{"class_name": "Dropout", "name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
ф

jkernel
kbias
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
–__call__
+—&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer≥{"class_name": "Dense", "name": "dense_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_26", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
к
ptrainable_variables
qregularization_losses
r	variables
s	keras_api
“__call__
+”&call_and_return_all_conditional_losses"ў
_tf_keras_layerњ{"class_name": "Dropout", "name": "dropout_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_23", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
х

tkernel
ubias
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"ќ
_tf_keras_layerі{"class_name": "Dense", "name": "dense_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_27", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
щ
	zdecay
{learning_rate
|momentum
}iter momentumС!momentumТ*momentumУ+momentumФ9momentumХ:momentumЦBmomentumЧCmomentumШ`momentumЩamomentumЪjmomentumЫkmomentumЬtmomentumЭumomentumЮ~momentumЯmomentum†Аmomentum°БmomentumҐВmomentum£Гmomentum§Дmomentum•Еmomentum¶ЖmomentumІЗmomentum®Иmomentum©Йmomentum™КmomentumЂЛmomentumђМmomentum≠НmomentumЃОmomentumѓПmomentum∞"
	optimizer
¶
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
А10
Б11
В12
Г13
Д14
Е15
Ж16
З17
И18
Й19
К20
Л21
М22
Н23
О24
П25
`26
a27
j28
k29
t30
u31"
trackable_list_wrapper
 "
trackable_list_wrapper
∆
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
А14
Б15
В16
Г17
Д18
Е19
Ж20
З21
И22
Й23
К24
Л25
М26
Н27
О28
П29
`30
a31
j32
k33
t34
u35"
trackable_list_wrapper
”
trainable_variables
Рnon_trainable_variables
Сmetrics
Тlayers
regularization_losses
Уlayer_metrics
 Фlayer_regularization_losses
	variables
±__call__
≥_default_save_signature
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
-
÷serving_default"
signature_map
і
~
embeddings
Хtrainable_variables
Цregularization_losses
Ч	variables
Ш	keras_api
„__call__
+Ў&call_and_return_all_conditional_losses"П
_tf_keras_layerх{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 4, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20000]}}
±

embeddings
Щtrainable_variables
Ъregularization_losses
Ы	variables
Ь	keras_api
ў__call__
+Џ&call_and_return_all_conditional_losses"М
_tf_keras_layerт{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 20000, "output_dim": 32, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
µ
Эmetrics
Юnon_trainable_variables
Яlayers
trainable_variables
regularization_losses
†layer_metrics
 °layer_regularization_losses
	variables
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
%:#  2conv1d_6/kernel
: 2conv1d_6/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
µ
Ґmetrics
£non_trainable_variables
§layers
"trainable_variables
#regularization_losses
•layer_metrics
 ¶layer_regularization_losses
$	variables
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Іmetrics
®non_trainable_variables
©layers
&trainable_variables
'regularization_losses
™layer_metrics
 Ђlayer_regularization_losses
(	variables
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
%:#	  2conv1d_7/kernel
: 2conv1d_7/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
µ
ђmetrics
≠non_trainable_variables
Ѓlayers
,trainable_variables
-regularization_losses
ѓlayer_metrics
 ∞layer_regularization_losses
.	variables
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±metrics
≤non_trainable_variables
≥layers
0trainable_variables
1regularization_losses
іlayer_metrics
 µlayer_regularization_losses
2	variables
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ґmetrics
Јnon_trainable_variables
Єlayers
4trainable_variables
5regularization_losses
єlayer_metrics
 Їlayer_regularization_losses
6	variables
Њ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_6/gamma
(:& 2batch_normalization_6/beta
1:/  (2!batch_normalization_6/moving_mean
5:3  (2%batch_normalization_6/moving_variance
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
90
:1
;2
<3"
trackable_list_wrapper
µ
їmetrics
Љnon_trainable_variables
љlayers
=trainable_variables
>regularization_losses
Њlayer_metrics
 њlayer_regularization_losses
?	variables
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
µ
јmetrics
Ѕnon_trainable_variables
¬layers
Ftrainable_variables
Gregularization_losses
√layer_metrics
 ƒlayer_regularization_losses
H	variables
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
≈metrics
∆non_trainable_variables
«layers
Jtrainable_variables
Kregularization_losses
»layer_metrics
 …layer_regularization_losses
L	variables
ƒ__call__
+≈&call_and_return_all_conditional_losses
'≈"call_and_return_conditional_losses"
_generic_user_object
И
 _query_dense
Ћ
_key_dense
ћ_value_dense
Ќ_softmax
ќ_dropout_layer
ѕ_output_dense
–trainable_variables
—regularization_losses
“	variables
”	keras_api
џ__call__
+№&call_and_return_all_conditional_losses"Д
_tf_keras_layerк{"class_name": "MultiHeadAttention", "name": "multi_head_attention_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "multi_head_attention_7", "trainable": true, "dtype": "float32", "num_heads": 1, "key_dim": 32, "value_dim": 32, "dropout": 0.0, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}
ѓ
‘layer_with_weights-0
‘layer-0
’layer_with_weights-1
’layer-1
÷trainable_variables
„regularization_losses
Ў	variables
ў	keras_api
Ё__call__
+ё&call_and_return_all_conditional_losses"»
_tf_keras_sequential©{"class_name": "Sequential", "name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 66, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 66, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_23_input"}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
м
	Џaxis

Мgamma
	Нbeta
џtrainable_variables
№regularization_losses
Ё	variables
ё	keras_api
я__call__
+а&call_and_return_all_conditional_losses"µ
_tf_keras_layerЫ{"class_name": "LayerNormalization", "name": "layer_normalization_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_14", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
м
	яaxis

Оgamma
	Пbeta
аtrainable_variables
бregularization_losses
в	variables
г	keras_api
б__call__
+в&call_and_return_all_conditional_losses"µ
_tf_keras_layerЫ{"class_name": "LayerNormalization", "name": "layer_normalization_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_15", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
н
дtrainable_variables
еregularization_losses
ж	variables
з	keras_api
г__call__
+д&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
н
иtrainable_variables
йregularization_losses
к	variables
л	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
¶
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7
И8
Й9
К10
Л11
М12
Н13
О14
П15"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7
И8
Й9
К10
Л11
М12
Н13
О14
П15"
trackable_list_wrapper
µ
мmetrics
нnon_trainable_variables
оlayers
Ttrainable_variables
Uregularization_losses
пlayer_metrics
 рlayer_regularization_losses
V	variables
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
сmetrics
тnon_trainable_variables
уlayers
Xtrainable_variables
Yregularization_losses
фlayer_metrics
 хlayer_regularization_losses
Z	variables
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
цmetrics
чnon_trainable_variables
шlayers
\trainable_variables
]regularization_losses
щlayer_metrics
 ъlayer_regularization_losses
^	variables
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
": 	»@2dense_25/kernel
:@2dense_25/bias
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
µ
ыmetrics
ьnon_trainable_variables
эlayers
btrainable_variables
cregularization_losses
юlayer_metrics
 €layer_regularization_losses
d	variables
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Аmetrics
Бnon_trainable_variables
Вlayers
ftrainable_variables
gregularization_losses
Гlayer_metrics
 Дlayer_regularization_losses
h	variables
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_26/kernel
:@2dense_26/bias
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
µ
Еmetrics
Жnon_trainable_variables
Зlayers
ltrainable_variables
mregularization_losses
Иlayer_metrics
 Йlayer_regularization_losses
n	variables
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Кmetrics
Лnon_trainable_variables
Мlayers
ptrainable_variables
qregularization_losses
Нlayer_metrics
 Оlayer_regularization_losses
r	variables
“__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_27/kernel
:2dense_27/bias
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
µ
Пmetrics
Рnon_trainable_variables
Сlayers
vtrainable_variables
wregularization_losses
Тlayer_metrics
 Уlayer_regularization_losses
x	variables
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
: (2decay
: (2learning_rate
: (2momentum
:	 (2SGD/iter
G:E 25token_and_position_embedding_3/embedding_6/embeddings
I:G
†Ь 25token_and_position_embedding_3/embedding_7/embeddings
M:K  27transformer_block_7/multi_head_attention_7/query/kernel
G:E 25transformer_block_7/multi_head_attention_7/query/bias
K:I  25transformer_block_7/multi_head_attention_7/key/kernel
E:C 23transformer_block_7/multi_head_attention_7/key/bias
M:K  27transformer_block_7/multi_head_attention_7/value/kernel
G:E 25transformer_block_7/multi_head_attention_7/value/bias
X:V  2Btransformer_block_7/multi_head_attention_7/attention_output/kernel
N:L 2@transformer_block_7/multi_head_attention_7/attention_output/bias
!: @2dense_23/kernel
:@2dense_23/bias
!:@ 2dense_24/kernel
: 2dense_24/bias
>:< 20transformer_block_7/layer_normalization_14/gamma
=:; 2/transformer_block_7/layer_normalization_14/beta
>:< 20transformer_block_7/layer_normalization_15/gamma
=:; 2/transformer_block_7/layer_normalization_15/beta
<
;0
<1
D2
E3"
trackable_list_wrapper
(
Ф0"
trackable_list_wrapper
Ѓ
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
Є
Хmetrics
Цnon_trainable_variables
Чlayers
Хtrainable_variables
Цregularization_losses
Шlayer_metrics
 Щlayer_regularization_losses
Ч	variables
„__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
Є
Ъmetrics
Ыnon_trainable_variables
Ьlayers
Щtrainable_variables
Ъregularization_losses
Эlayer_metrics
 Юlayer_regularization_losses
Ы	variables
ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
Ћ
Яpartial_output_shape
†full_output_shape
Аkernel
	Бbias
°trainable_variables
Ґregularization_losses
£	variables
§	keras_api
з__call__
+и&call_and_return_all_conditional_losses"л
_tf_keras_layer—{"class_name": "EinsumDense", "name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
«
•partial_output_shape
¶full_output_shape
Вkernel
	Гbias
Іtrainable_variables
®regularization_losses
©	variables
™	keras_api
й__call__
+к&call_and_return_all_conditional_losses"з
_tf_keras_layerЌ{"class_name": "EinsumDense", "name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
Ћ
Ђpartial_output_shape
ђfull_output_shape
Дkernel
	Еbias
≠trainable_variables
Ѓregularization_losses
ѓ	variables
∞	keras_api
л__call__
+м&call_and_return_all_conditional_losses"л
_tf_keras_layer—{"class_name": "EinsumDense", "name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 1, 32], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
л
±trainable_variables
≤regularization_losses
≥	variables
і	keras_api
н__call__
+о&call_and_return_all_conditional_losses"÷
_tf_keras_layerЉ{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}}
з
µtrainable_variables
ґregularization_losses
Ј	variables
Є	keras_api
п__call__
+р&call_and_return_all_conditional_losses"“
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
а
єpartial_output_shape
Їfull_output_shape
Жkernel
	Зbias
їtrainable_variables
Љregularization_losses
љ	variables
Њ	keras_api
с__call__
+т&call_and_return_all_conditional_losses"А
_tf_keras_layerж{"class_name": "EinsumDense", "name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 32], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 1, 32]}}
`
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7"
trackable_list_wrapper
Є
њmetrics
јnon_trainable_variables
Ѕlayers
–trainable_variables
—regularization_losses
¬layer_metrics
 √layer_regularization_losses
“	variables
џ__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
ю
Иkernel
	Йbias
ƒtrainable_variables
≈regularization_losses
∆	variables
«	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"—
_tf_keras_layerЈ{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 32]}}
А
Кkernel
	Лbias
»trainable_variables
…regularization_losses
 	variables
Ћ	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 64]}}
@
И0
Й1
К2
Л3"
trackable_list_wrapper
 "
trackable_list_wrapper
@
И0
Й1
К2
Л3"
trackable_list_wrapper
Є
÷trainable_variables
ћnon_trainable_variables
Ќmetrics
ќlayers
„regularization_losses
ѕlayer_metrics
 –layer_regularization_losses
Ў	variables
Ё__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
Є
—metrics
“non_trainable_variables
”layers
џtrainable_variables
№regularization_losses
‘layer_metrics
 ’layer_regularization_losses
Ё	variables
я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
Є
÷metrics
„non_trainable_variables
Ўlayers
аtrainable_variables
бregularization_losses
ўlayer_metrics
 Џlayer_regularization_losses
в	variables
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
џmetrics
№non_trainable_variables
Ёlayers
дtrainable_variables
еregularization_losses
ёlayer_metrics
 яlayer_regularization_losses
ж	variables
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
аmetrics
бnon_trainable_variables
вlayers
иtrainable_variables
йregularization_losses
гlayer_metrics
 дlayer_regularization_losses
к	variables
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
N0
O1
P2
Q3
R4
S5"
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
њ

еtotal

жcount
з	variables
и	keras_api"Д
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
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
Є
йmetrics
кnon_trainable_variables
лlayers
°trainable_variables
Ґregularization_losses
мlayer_metrics
 нlayer_regularization_losses
£	variables
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
Є
оmetrics
пnon_trainable_variables
рlayers
Іtrainable_variables
®regularization_losses
сlayer_metrics
 тlayer_regularization_losses
©	variables
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
Є
уmetrics
фnon_trainable_variables
хlayers
≠trainable_variables
Ѓregularization_losses
цlayer_metrics
 чlayer_regularization_losses
ѓ	variables
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
шmetrics
щnon_trainable_variables
ъlayers
±trainable_variables
≤regularization_losses
ыlayer_metrics
 ьlayer_regularization_losses
≥	variables
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
эmetrics
юnon_trainable_variables
€layers
µtrainable_variables
ґregularization_losses
Аlayer_metrics
 Бlayer_regularization_losses
Ј	variables
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ж0
З1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ж0
З1"
trackable_list_wrapper
Є
Вmetrics
Гnon_trainable_variables
Дlayers
їtrainable_variables
Љregularization_losses
Еlayer_metrics
 Жlayer_regularization_losses
љ	variables
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
P
 0
Ћ1
ћ2
Ќ3
ќ4
ѕ5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
Є
Зmetrics
Иnon_trainable_variables
Йlayers
ƒtrainable_variables
≈regularization_losses
Кlayer_metrics
 Лlayer_regularization_losses
∆	variables
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
Є
Мmetrics
Нnon_trainable_variables
Оlayers
»trainable_variables
…regularization_losses
Пlayer_metrics
 Рlayer_regularization_losses
 	variables
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
‘0
’1"
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
:  (2total
:  (2count
0
е0
ж1"
trackable_list_wrapper
.
з	variables"
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
0:.  2SGD/conv1d_6/kernel/momentum
&:$ 2SGD/conv1d_6/bias/momentum
0:.	  2SGD/conv1d_7/kernel/momentum
&:$ 2SGD/conv1d_7/bias/momentum
4:2 2(SGD/batch_normalization_6/gamma/momentum
3:1 2'SGD/batch_normalization_6/beta/momentum
4:2 2(SGD/batch_normalization_7/gamma/momentum
3:1 2'SGD/batch_normalization_7/beta/momentum
-:+	»@2SGD/dense_25/kernel/momentum
&:$@2SGD/dense_25/bias/momentum
,:*@@2SGD/dense_26/kernel/momentum
&:$@2SGD/dense_26/bias/momentum
,:*@2SGD/dense_27/kernel/momentum
&:$2SGD/dense_27/bias/momentum
R:P 2BSGD/token_and_position_embedding_3/embedding_6/embeddings/momentum
T:R
†Ь 2BSGD/token_and_position_embedding_3/embedding_7/embeddings/momentum
X:V  2DSGD/transformer_block_7/multi_head_attention_7/query/kernel/momentum
R:P 2BSGD/transformer_block_7/multi_head_attention_7/query/bias/momentum
V:T  2BSGD/transformer_block_7/multi_head_attention_7/key/kernel/momentum
P:N 2@SGD/transformer_block_7/multi_head_attention_7/key/bias/momentum
X:V  2DSGD/transformer_block_7/multi_head_attention_7/value/kernel/momentum
R:P 2BSGD/transformer_block_7/multi_head_attention_7/value/bias/momentum
c:a  2OSGD/transformer_block_7/multi_head_attention_7/attention_output/kernel/momentum
Y:W 2MSGD/transformer_block_7/multi_head_attention_7/attention_output/bias/momentum
,:* @2SGD/dense_23/kernel/momentum
&:$@2SGD/dense_23/bias/momentum
,:*@ 2SGD/dense_24/kernel/momentum
&:$ 2SGD/dense_24/bias/momentum
I:G 2=SGD/transformer_block_7/layer_normalization_14/gamma/momentum
H:F 2<SGD/transformer_block_7/layer_normalization_14/beta/momentum
I:G 2=SGD/transformer_block_7/layer_normalization_15/gamma/momentum
H:F 2<SGD/transformer_block_7/layer_normalization_15/beta/momentum
о2л
(__inference_model_3_layer_call_fn_411089
(__inference_model_3_layer_call_fn_411806
(__inference_model_3_layer_call_fn_410917
(__inference_model_3_layer_call_fn_411884ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
C__inference_model_3_layer_call_and_return_conditional_losses_411728
C__inference_model_3_layer_call_and_return_conditional_losses_411485
C__inference_model_3_layer_call_and_return_conditional_losses_410744
C__inference_model_3_layer_call_and_return_conditional_losses_410650ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Й2Ж
!__inference__wrapped_model_409294а
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *PҐM
KЪH
#К 
input_7€€€€€€€€€†Ь
!К
input_8€€€€€€€€€
д2б
?__inference_token_and_position_embedding_3_layer_call_fn_411917Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€2ь
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_411908Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_conv1d_6_layer_call_fn_411942Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv1d_6_layer_call_and_return_conditional_losses_411933Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
П2М
4__inference_average_pooling1d_9_layer_call_fn_409309”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™2І
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_409303”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
”2–
)__inference_conv1d_7_layer_call_fn_411967Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_conv1d_7_layer_call_and_return_conditional_losses_411958Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Р2Н
5__inference_average_pooling1d_10_layer_call_fn_409324”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ2®
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_409318”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Р2Н
5__inference_average_pooling1d_11_layer_call_fn_409339”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ2®
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_409333”
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *3Ґ0
.К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ2Ч
6__inference_batch_normalization_6_layer_call_fn_412036
6__inference_batch_normalization_6_layer_call_fn_412049
6__inference_batch_normalization_6_layer_call_fn_412118
6__inference_batch_normalization_6_layer_call_fn_412131і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412085
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412003
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412023
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412105і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ъ2Ч
6__inference_batch_normalization_7_layer_call_fn_412282
6__inference_batch_normalization_7_layer_call_fn_412213
6__inference_batch_normalization_7_layer_call_fn_412295
6__inference_batch_normalization_7_layer_call_fn_412200і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ж2Г
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412167
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412249
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412187
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412269і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_add_3_layer_call_fn_412307Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_add_3_layer_call_and_return_conditional_losses_412301Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ґ2Я
4__inference_transformer_block_7_layer_call_fn_412619
4__inference_transformer_block_7_layer_call_fn_412656∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ў2’
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412582
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412455∞
І≤£
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
‘2—
*__inference_flatten_3_layer_call_fn_412667Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_flatten_3_layer_call_and_return_conditional_losses_412662Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_concatenate_3_layer_call_fn_412680Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_concatenate_3_layer_call_and_return_conditional_losses_412674Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_25_layer_call_fn_412700Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_25_layer_call_and_return_conditional_losses_412691Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ф2С
+__inference_dropout_22_layer_call_fn_412727
+__inference_dropout_22_layer_call_fn_412722і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 2«
F__inference_dropout_22_layer_call_and_return_conditional_losses_412712
F__inference_dropout_22_layer_call_and_return_conditional_losses_412717і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_dense_26_layer_call_fn_412747Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_26_layer_call_and_return_conditional_losses_412738Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ф2С
+__inference_dropout_23_layer_call_fn_412769
+__inference_dropout_23_layer_call_fn_412774і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 2«
F__inference_dropout_23_layer_call_and_return_conditional_losses_412764
F__inference_dropout_23_layer_call_and_return_conditional_losses_412759і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_dense_27_layer_call_fn_412793Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_27_layer_call_and_return_conditional_losses_412784Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“Bѕ
$__inference_signature_wrapper_411175input_7input_8"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€
-__inference_sequential_7_layer_call_fn_409786
-__inference_sequential_7_layer_call_fn_412920
-__inference_sequential_7_layer_call_fn_409759
-__inference_sequential_7_layer_call_fn_412933ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
о2л
H__inference_sequential_7_layer_call_and_return_conditional_losses_409731
H__inference_sequential_7_layer_call_and_return_conditional_losses_412850
H__inference_sequential_7_layer_call_and_return_conditional_losses_409717
H__inference_sequential_7_layer_call_and_return_conditional_losses_412907ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_23_layer_call_fn_412973Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_23_layer_call_and_return_conditional_losses_412964Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_24_layer_call_fn_413012Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_24_layer_call_and_return_conditional_losses_413003Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 н
!__inference__wrapped_model_409294«4~ !*+<9;:EBDCАБВГДЕЖЗМНИЙКЛОП`ajktuZҐW
PҐM
KЪH
#К 
input_7€€€€€€€€€†Ь
!К
input_8€€€€€€€€€
™ "3™0
.
dense_27"К
dense_27€€€€€€€€€’
A__inference_add_3_layer_call_and_return_conditional_losses_412301ПbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€B 
&К#
inputs/1€€€€€€€€€B 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ ≠
&__inference_add_3_layer_call_fn_412307ВbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€B 
&К#
inputs/1€€€€€€€€€B 
™ "К€€€€€€€€€B ў
P__inference_average_pooling1d_10_layer_call_and_return_conditional_losses_409318ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∞
5__inference_average_pooling1d_10_layer_call_fn_409324wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€ў
P__inference_average_pooling1d_11_layer_call_and_return_conditional_losses_409333ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ∞
5__inference_average_pooling1d_11_layer_call_fn_409339wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Ў
O__inference_average_pooling1d_9_layer_call_and_return_conditional_losses_409303ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ѓ
4__inference_average_pooling1d_9_layer_call_fn_409309wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€њ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412003j;<9:7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ њ
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412023j<9;:7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ —
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412085|;<9:@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ —
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_412105|<9;:@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ Ч
6__inference_batch_normalization_6_layer_call_fn_412036];<9:7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p
™ "К€€€€€€€€€B Ч
6__inference_batch_normalization_6_layer_call_fn_412049]<9;:7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p 
™ "К€€€€€€€€€B ©
6__inference_batch_normalization_6_layer_call_fn_412118o;<9:@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "%К"€€€€€€€€€€€€€€€€€€ ©
6__inference_batch_normalization_6_layer_call_fn_412131o<9;:@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "%К"€€€€€€€€€€€€€€€€€€ —
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412167|DEBC@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ —
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412187|EBDC@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ њ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412249jDEBC7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ њ
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_412269jEBDC7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ ©
6__inference_batch_normalization_7_layer_call_fn_412200oDEBC@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p
™ "%К"€€€€€€€€€€€€€€€€€€ ©
6__inference_batch_normalization_7_layer_call_fn_412213oEBDC@Ґ=
6Ґ3
-К*
inputs€€€€€€€€€€€€€€€€€€ 
p 
™ "%К"€€€€€€€€€€€€€€€€€€ Ч
6__inference_batch_normalization_7_layer_call_fn_412282]DEBC7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p
™ "К€€€€€€€€€B Ч
6__inference_batch_normalization_7_layer_call_fn_412295]EBDC7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p 
™ "К€€€€€€€€€B ”
I__inference_concatenate_3_layer_call_and_return_conditional_losses_412674Е[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€ј
"К
inputs/1€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€»
Ъ ™
.__inference_concatenate_3_layer_call_fn_412680x[ҐX
QҐN
LЪI
#К 
inputs/0€€€€€€€€€ј
"К
inputs/1€€€€€€€€€
™ "К€€€€€€€€€»∞
D__inference_conv1d_6_layer_call_and_return_conditional_losses_411933h !5Ґ2
+Ґ(
&К#
inputs€€€€€€€€€†Ь 
™ "+Ґ(
!К
0€€€€€€€€€†Ь 
Ъ И
)__inference_conv1d_6_layer_call_fn_411942[ !5Ґ2
+Ґ(
&К#
inputs€€€€€€€€€†Ь 
™ "К€€€€€€€€€†Ь Ѓ
D__inference_conv1d_7_layer_call_and_return_conditional_losses_411958f*+4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Ъ 
™ "*Ґ'
 К
0€€€€€€€€€Ъ 
Ъ Ж
)__inference_conv1d_7_layer_call_fn_411967Y*+4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Ъ 
™ "К€€€€€€€€€Ъ Ѓ
D__inference_dense_23_layer_call_and_return_conditional_losses_412964fИЙ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€B 
™ ")Ґ&
К
0€€€€€€€€€B@
Ъ Ж
)__inference_dense_23_layer_call_fn_412973YИЙ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€B 
™ "К€€€€€€€€€B@Ѓ
D__inference_dense_24_layer_call_and_return_conditional_losses_413003fКЛ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€B@
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ Ж
)__inference_dense_24_layer_call_fn_413012YКЛ3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€B@
™ "К€€€€€€€€€B •
D__inference_dense_25_layer_call_and_return_conditional_losses_412691]`a0Ґ-
&Ґ#
!К
inputs€€€€€€€€€»
™ "%Ґ"
К
0€€€€€€€€€@
Ъ }
)__inference_dense_25_layer_call_fn_412700P`a0Ґ-
&Ґ#
!К
inputs€€€€€€€€€»
™ "К€€€€€€€€€@§
D__inference_dense_26_layer_call_and_return_conditional_losses_412738\jk/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ |
)__inference_dense_26_layer_call_fn_412747Ojk/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@§
D__inference_dense_27_layer_call_and_return_conditional_losses_412784\tu/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_27_layer_call_fn_412793Otu/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€¶
F__inference_dropout_22_layer_call_and_return_conditional_losses_412712\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¶
F__inference_dropout_22_layer_call_and_return_conditional_losses_412717\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ~
+__inference_dropout_22_layer_call_fn_412722O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@~
+__inference_dropout_22_layer_call_fn_412727O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@¶
F__inference_dropout_23_layer_call_and_return_conditional_losses_412759\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ¶
F__inference_dropout_23_layer_call_and_return_conditional_losses_412764\3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ~
+__inference_dropout_23_layer_call_fn_412769O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p
™ "К€€€€€€€€€@~
+__inference_dropout_23_layer_call_fn_412774O3Ґ0
)Ґ&
 К
inputs€€€€€€€€€@
p 
™ "К€€€€€€€€€@¶
E__inference_flatten_3_layer_call_and_return_conditional_losses_412662]3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€B 
™ "&Ґ#
К
0€€€€€€€€€ј
Ъ ~
*__inference_flatten_3_layer_call_fn_412667P3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€B 
™ "К€€€€€€€€€јЙ
C__inference_model_3_layer_call_and_return_conditional_losses_410650Ѕ4~ !*+;<9:DEBCАБВГДЕЖЗМНИЙКЛОП`ajktubҐ_
XҐU
KЪH
#К 
input_7€€€€€€€€€†Ь
!К
input_8€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Й
C__inference_model_3_layer_call_and_return_conditional_losses_410744Ѕ4~ !*+<9;:EBDCАБВГДЕЖЗМНИЙКЛОП`ajktubҐ_
XҐU
KЪH
#К 
input_7€€€€€€€€€†Ь
!К
input_8€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Л
C__inference_model_3_layer_call_and_return_conditional_losses_411485√4~ !*+;<9:DEBCАБВГДЕЖЗМНИЙКЛОП`ajktudҐa
ZҐW
MЪJ
$К!
inputs/0€€€€€€€€€†Ь
"К
inputs/1€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Л
C__inference_model_3_layer_call_and_return_conditional_losses_411728√4~ !*+<9;:EBDCАБВГДЕЖЗМНИЙКЛОП`ajktudҐa
ZҐW
MЪJ
$К!
inputs/0€€€€€€€€€†Ь
"К
inputs/1€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ б
(__inference_model_3_layer_call_fn_410917і4~ !*+;<9:DEBCАБВГДЕЖЗМНИЙКЛОП`ajktubҐ_
XҐU
KЪH
#К 
input_7€€€€€€€€€†Ь
!К
input_8€€€€€€€€€
p

 
™ "К€€€€€€€€€б
(__inference_model_3_layer_call_fn_411089і4~ !*+<9;:EBDCАБВГДЕЖЗМНИЙКЛОП`ajktubҐ_
XҐU
KЪH
#К 
input_7€€€€€€€€€†Ь
!К
input_8€€€€€€€€€
p 

 
™ "К€€€€€€€€€г
(__inference_model_3_layer_call_fn_411806ґ4~ !*+;<9:DEBCАБВГДЕЖЗМНИЙКЛОП`ajktudҐa
ZҐW
MЪJ
$К!
inputs/0€€€€€€€€€†Ь
"К
inputs/1€€€€€€€€€
p

 
™ "К€€€€€€€€€г
(__inference_model_3_layer_call_fn_411884ґ4~ !*+<9;:EBDCАБВГДЕЖЗМНИЙКЛОП`ajktudҐa
ZҐW
MЪJ
$К!
inputs/0€€€€€€€€€†Ь
"К
inputs/1€€€€€€€€€
p 

 
™ "К€€€€€€€€€∆
H__inference_sequential_7_layer_call_and_return_conditional_losses_409717zИЙКЛCҐ@
9Ґ6
,К)
dense_23_input€€€€€€€€€B 
p

 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ ∆
H__inference_sequential_7_layer_call_and_return_conditional_losses_409731zИЙКЛCҐ@
9Ґ6
,К)
dense_23_input€€€€€€€€€B 
p 

 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ Њ
H__inference_sequential_7_layer_call_and_return_conditional_losses_412850rИЙКЛ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€B 
p

 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ Њ
H__inference_sequential_7_layer_call_and_return_conditional_losses_412907rИЙКЛ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€B 
p 

 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ Ю
-__inference_sequential_7_layer_call_fn_409759mИЙКЛCҐ@
9Ґ6
,К)
dense_23_input€€€€€€€€€B 
p

 
™ "К€€€€€€€€€B Ю
-__inference_sequential_7_layer_call_fn_409786mИЙКЛCҐ@
9Ґ6
,К)
dense_23_input€€€€€€€€€B 
p 

 
™ "К€€€€€€€€€B Ц
-__inference_sequential_7_layer_call_fn_412920eИЙКЛ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€B 
p

 
™ "К€€€€€€€€€B Ц
-__inference_sequential_7_layer_call_fn_412933eИЙКЛ;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€B 
p 

 
™ "К€€€€€€€€€B Б
$__inference_signature_wrapper_411175Ў4~ !*+<9;:EBDCАБВГДЕЖЗМНИЙКЛОП`ajktukҐh
Ґ 
a™^
.
input_7#К 
input_7€€€€€€€€€†Ь
,
input_8!К
input_8€€€€€€€€€"3™0
.
dense_27"К
dense_27€€€€€€€€€љ
Z__inference_token_and_position_embedding_3_layer_call_and_return_conditional_losses_411908_~,Ґ)
"Ґ
К
x€€€€€€€€€†Ь
™ "+Ґ(
!К
0€€€€€€€€€†Ь 
Ъ Х
?__inference_token_and_position_embedding_3_layer_call_fn_411917R~,Ґ)
"Ґ
К
x€€€€€€€€€†Ь
™ "К€€€€€€€€€†Ь Џ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412455Ж АБВГДЕЖЗМНИЙКЛОП7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ Џ
O__inference_transformer_block_7_layer_call_and_return_conditional_losses_412582Ж АБВГДЕЖЗМНИЙКЛОП7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p 
™ ")Ґ&
К
0€€€€€€€€€B 
Ъ ±
4__inference_transformer_block_7_layer_call_fn_412619y АБВГДЕЖЗМНИЙКЛОП7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p
™ "К€€€€€€€€€B ±
4__inference_transformer_block_7_layer_call_fn_412656y АБВГДЕЖЗМНИЙКЛОП7Ґ4
-Ґ*
$К!
inputs€€€€€€€€€B 
p 
™ "К€€€€€€€€€B 