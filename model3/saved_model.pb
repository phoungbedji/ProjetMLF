??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.1-0-g85c8b2a817f8??
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:Z*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:Z*
dtype0
`
UzVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUz
Y
Uz/Read/ReadVariableOpReadVariableOpUz*
_output_shapes

:Z*
dtype0
`
UgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUg
Y
Ug/Read/ReadVariableOpReadVariableOpUg*
_output_shapes

:Z*
dtype0
`
UrVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUr
Y
Ur/Read/ReadVariableOpReadVariableOpUr*
_output_shapes

:Z*
dtype0
`
UhVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUh
Y
Uh/Read/ReadVariableOpReadVariableOpUh*
_output_shapes

:Z*
dtype0
`
WzVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWz
Y
Wz/Read/ReadVariableOpReadVariableOpWz*
_output_shapes

:ZZ*
dtype0
`
WgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWg
Y
Wg/Read/ReadVariableOpReadVariableOpWg*
_output_shapes

:ZZ*
dtype0
`
WrVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWr
Y
Wr/Read/ReadVariableOpReadVariableOpWr*
_output_shapes

:ZZ*
dtype0
`
WhVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWh
Y
Wh/Read/ReadVariableOpReadVariableOpWh*
_output_shapes

:ZZ*
dtype0
\
bzVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebz
U
bz/Read/ReadVariableOpReadVariableOpbz*
_output_shapes
:Z*
dtype0
\
bgVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebg
U
bg/Read/ReadVariableOpReadVariableOpbg*
_output_shapes
:Z*
dtype0
\
brVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebr
U
br/Read/ReadVariableOpReadVariableOpbr*
_output_shapes
:Z*
dtype0
\
bhVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebh
U
bh/Read/ReadVariableOpReadVariableOpbh*
_output_shapes
:Z*
dtype0
d
Uz_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUz_1
]
Uz_1/Read/ReadVariableOpReadVariableOpUz_1*
_output_shapes

:Z*
dtype0
d
Ug_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUg_1
]
Ug_1/Read/ReadVariableOpReadVariableOpUg_1*
_output_shapes

:Z*
dtype0
d
Ur_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUr_1
]
Ur_1/Read/ReadVariableOpReadVariableOpUr_1*
_output_shapes

:Z*
dtype0
d
Uh_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUh_1
]
Uh_1/Read/ReadVariableOpReadVariableOpUh_1*
_output_shapes

:Z*
dtype0
d
Wz_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWz_1
]
Wz_1/Read/ReadVariableOpReadVariableOpWz_1*
_output_shapes

:ZZ*
dtype0
d
Wg_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWg_1
]
Wg_1/Read/ReadVariableOpReadVariableOpWg_1*
_output_shapes

:ZZ*
dtype0
d
Wr_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWr_1
]
Wr_1/Read/ReadVariableOpReadVariableOpWr_1*
_output_shapes

:ZZ*
dtype0
d
Wh_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWh_1
]
Wh_1/Read/ReadVariableOpReadVariableOpWh_1*
_output_shapes

:ZZ*
dtype0
`
bz_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebz_1
Y
bz_1/Read/ReadVariableOpReadVariableOpbz_1*
_output_shapes
:Z*
dtype0
`
bg_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebg_1
Y
bg_1/Read/ReadVariableOpReadVariableOpbg_1*
_output_shapes
:Z*
dtype0
`
br_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebr_1
Y
br_1/Read/ReadVariableOpReadVariableOpbr_1*
_output_shapes
:Z*
dtype0
`
bh_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebh_1
Y
bh_1/Read/ReadVariableOpReadVariableOpbh_1*
_output_shapes
:Z*
dtype0
d
Uz_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUz_2
]
Uz_2/Read/ReadVariableOpReadVariableOpUz_2*
_output_shapes

:Z*
dtype0
d
Ug_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUg_2
]
Ug_2/Read/ReadVariableOpReadVariableOpUg_2*
_output_shapes

:Z*
dtype0
d
Ur_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUr_2
]
Ur_2/Read/ReadVariableOpReadVariableOpUr_2*
_output_shapes

:Z*
dtype0
d
Uh_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameUh_2
]
Uh_2/Read/ReadVariableOpReadVariableOpUh_2*
_output_shapes

:Z*
dtype0
d
Wz_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWz_2
]
Wz_2/Read/ReadVariableOpReadVariableOpWz_2*
_output_shapes

:ZZ*
dtype0
d
Wg_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWg_2
]
Wg_2/Read/ReadVariableOpReadVariableOpWg_2*
_output_shapes

:ZZ*
dtype0
d
Wr_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWr_2
]
Wr_2/Read/ReadVariableOpReadVariableOpWr_2*
_output_shapes

:ZZ*
dtype0
d
Wh_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameWh_2
]
Wh_2/Read/ReadVariableOpReadVariableOpWh_2*
_output_shapes

:ZZ*
dtype0
`
bz_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebz_2
Y
bz_2/Read/ReadVariableOpReadVariableOpbz_2*
_output_shapes
:Z*
dtype0
`
bg_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebg_2
Y
bg_2/Read/ReadVariableOpReadVariableOpbg_2*
_output_shapes
:Z*
dtype0
`
br_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebr_2
Y
br_2/Read/ReadVariableOpReadVariableOpbr_2*
_output_shapes
:Z*
dtype0
`
bh_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namebh_2
Y
bh_2/Read/ReadVariableOpReadVariableOpbh_2*
_output_shapes
:Z*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:Z*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:Z*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:Z*
dtype0
n
	Adam/Uz/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Uz/m
g
Adam/Uz/m/Read/ReadVariableOpReadVariableOp	Adam/Uz/m*
_output_shapes

:Z*
dtype0
n
	Adam/Ug/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Ug/m
g
Adam/Ug/m/Read/ReadVariableOpReadVariableOp	Adam/Ug/m*
_output_shapes

:Z*
dtype0
n
	Adam/Ur/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Ur/m
g
Adam/Ur/m/Read/ReadVariableOpReadVariableOp	Adam/Ur/m*
_output_shapes

:Z*
dtype0
n
	Adam/Uh/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Uh/m
g
Adam/Uh/m/Read/ReadVariableOpReadVariableOp	Adam/Uh/m*
_output_shapes

:Z*
dtype0
n
	Adam/Wz/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wz/m
g
Adam/Wz/m/Read/ReadVariableOpReadVariableOp	Adam/Wz/m*
_output_shapes

:ZZ*
dtype0
n
	Adam/Wg/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wg/m
g
Adam/Wg/m/Read/ReadVariableOpReadVariableOp	Adam/Wg/m*
_output_shapes

:ZZ*
dtype0
n
	Adam/Wr/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wr/m
g
Adam/Wr/m/Read/ReadVariableOpReadVariableOp	Adam/Wr/m*
_output_shapes

:ZZ*
dtype0
n
	Adam/Wh/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wh/m
g
Adam/Wh/m/Read/ReadVariableOpReadVariableOp	Adam/Wh/m*
_output_shapes

:ZZ*
dtype0
j
	Adam/bz/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/bz/m
c
Adam/bz/m/Read/ReadVariableOpReadVariableOp	Adam/bz/m*
_output_shapes
:Z*
dtype0
j
	Adam/bg/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/bg/m
c
Adam/bg/m/Read/ReadVariableOpReadVariableOp	Adam/bg/m*
_output_shapes
:Z*
dtype0
j
	Adam/br/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/br/m
c
Adam/br/m/Read/ReadVariableOpReadVariableOp	Adam/br/m*
_output_shapes
:Z*
dtype0
j
	Adam/bh/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/bh/m
c
Adam/bh/m/Read/ReadVariableOpReadVariableOp	Adam/bh/m*
_output_shapes
:Z*
dtype0
r
Adam/Uz/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uz/m_1
k
Adam/Uz/m_1/Read/ReadVariableOpReadVariableOpAdam/Uz/m_1*
_output_shapes

:Z*
dtype0
r
Adam/Ug/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ug/m_1
k
Adam/Ug/m_1/Read/ReadVariableOpReadVariableOpAdam/Ug/m_1*
_output_shapes

:Z*
dtype0
r
Adam/Ur/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ur/m_1
k
Adam/Ur/m_1/Read/ReadVariableOpReadVariableOpAdam/Ur/m_1*
_output_shapes

:Z*
dtype0
r
Adam/Uh/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uh/m_1
k
Adam/Uh/m_1/Read/ReadVariableOpReadVariableOpAdam/Uh/m_1*
_output_shapes

:Z*
dtype0
r
Adam/Wz/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wz/m_1
k
Adam/Wz/m_1/Read/ReadVariableOpReadVariableOpAdam/Wz/m_1*
_output_shapes

:ZZ*
dtype0
r
Adam/Wg/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wg/m_1
k
Adam/Wg/m_1/Read/ReadVariableOpReadVariableOpAdam/Wg/m_1*
_output_shapes

:ZZ*
dtype0
r
Adam/Wr/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wr/m_1
k
Adam/Wr/m_1/Read/ReadVariableOpReadVariableOpAdam/Wr/m_1*
_output_shapes

:ZZ*
dtype0
r
Adam/Wh/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wh/m_1
k
Adam/Wh/m_1/Read/ReadVariableOpReadVariableOpAdam/Wh/m_1*
_output_shapes

:ZZ*
dtype0
n
Adam/bz/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bz/m_1
g
Adam/bz/m_1/Read/ReadVariableOpReadVariableOpAdam/bz/m_1*
_output_shapes
:Z*
dtype0
n
Adam/bg/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bg/m_1
g
Adam/bg/m_1/Read/ReadVariableOpReadVariableOpAdam/bg/m_1*
_output_shapes
:Z*
dtype0
n
Adam/br/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/br/m_1
g
Adam/br/m_1/Read/ReadVariableOpReadVariableOpAdam/br/m_1*
_output_shapes
:Z*
dtype0
n
Adam/bh/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bh/m_1
g
Adam/bh/m_1/Read/ReadVariableOpReadVariableOpAdam/bh/m_1*
_output_shapes
:Z*
dtype0
r
Adam/Uz/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uz/m_2
k
Adam/Uz/m_2/Read/ReadVariableOpReadVariableOpAdam/Uz/m_2*
_output_shapes

:Z*
dtype0
r
Adam/Ug/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ug/m_2
k
Adam/Ug/m_2/Read/ReadVariableOpReadVariableOpAdam/Ug/m_2*
_output_shapes

:Z*
dtype0
r
Adam/Ur/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ur/m_2
k
Adam/Ur/m_2/Read/ReadVariableOpReadVariableOpAdam/Ur/m_2*
_output_shapes

:Z*
dtype0
r
Adam/Uh/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uh/m_2
k
Adam/Uh/m_2/Read/ReadVariableOpReadVariableOpAdam/Uh/m_2*
_output_shapes

:Z*
dtype0
r
Adam/Wz/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wz/m_2
k
Adam/Wz/m_2/Read/ReadVariableOpReadVariableOpAdam/Wz/m_2*
_output_shapes

:ZZ*
dtype0
r
Adam/Wg/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wg/m_2
k
Adam/Wg/m_2/Read/ReadVariableOpReadVariableOpAdam/Wg/m_2*
_output_shapes

:ZZ*
dtype0
r
Adam/Wr/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wr/m_2
k
Adam/Wr/m_2/Read/ReadVariableOpReadVariableOpAdam/Wr/m_2*
_output_shapes

:ZZ*
dtype0
r
Adam/Wh/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wh/m_2
k
Adam/Wh/m_2/Read/ReadVariableOpReadVariableOpAdam/Wh/m_2*
_output_shapes

:ZZ*
dtype0
n
Adam/bz/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bz/m_2
g
Adam/bz/m_2/Read/ReadVariableOpReadVariableOpAdam/bz/m_2*
_output_shapes
:Z*
dtype0
n
Adam/bg/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bg/m_2
g
Adam/bg/m_2/Read/ReadVariableOpReadVariableOpAdam/bg/m_2*
_output_shapes
:Z*
dtype0
n
Adam/br/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/br/m_2
g
Adam/br/m_2/Read/ReadVariableOpReadVariableOpAdam/br/m_2*
_output_shapes
:Z*
dtype0
n
Adam/bh/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bh/m_2
g
Adam/bh/m_2/Read/ReadVariableOpReadVariableOpAdam/bh/m_2*
_output_shapes
:Z*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:Z*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:Z*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:Z*
dtype0
n
	Adam/Uz/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Uz/v
g
Adam/Uz/v/Read/ReadVariableOpReadVariableOp	Adam/Uz/v*
_output_shapes

:Z*
dtype0
n
	Adam/Ug/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Ug/v
g
Adam/Ug/v/Read/ReadVariableOpReadVariableOp	Adam/Ug/v*
_output_shapes

:Z*
dtype0
n
	Adam/Ur/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Ur/v
g
Adam/Ur/v/Read/ReadVariableOpReadVariableOp	Adam/Ur/v*
_output_shapes

:Z*
dtype0
n
	Adam/Uh/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_name	Adam/Uh/v
g
Adam/Uh/v/Read/ReadVariableOpReadVariableOp	Adam/Uh/v*
_output_shapes

:Z*
dtype0
n
	Adam/Wz/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wz/v
g
Adam/Wz/v/Read/ReadVariableOpReadVariableOp	Adam/Wz/v*
_output_shapes

:ZZ*
dtype0
n
	Adam/Wg/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wg/v
g
Adam/Wg/v/Read/ReadVariableOpReadVariableOp	Adam/Wg/v*
_output_shapes

:ZZ*
dtype0
n
	Adam/Wr/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wr/v
g
Adam/Wr/v/Read/ReadVariableOpReadVariableOp	Adam/Wr/v*
_output_shapes

:ZZ*
dtype0
n
	Adam/Wh/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_name	Adam/Wh/v
g
Adam/Wh/v/Read/ReadVariableOpReadVariableOp	Adam/Wh/v*
_output_shapes

:ZZ*
dtype0
j
	Adam/bz/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/bz/v
c
Adam/bz/v/Read/ReadVariableOpReadVariableOp	Adam/bz/v*
_output_shapes
:Z*
dtype0
j
	Adam/bg/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/bg/v
c
Adam/bg/v/Read/ReadVariableOpReadVariableOp	Adam/bg/v*
_output_shapes
:Z*
dtype0
j
	Adam/br/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/br/v
c
Adam/br/v/Read/ReadVariableOpReadVariableOp	Adam/br/v*
_output_shapes
:Z*
dtype0
j
	Adam/bh/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_name	Adam/bh/v
c
Adam/bh/v/Read/ReadVariableOpReadVariableOp	Adam/bh/v*
_output_shapes
:Z*
dtype0
r
Adam/Uz/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uz/v_1
k
Adam/Uz/v_1/Read/ReadVariableOpReadVariableOpAdam/Uz/v_1*
_output_shapes

:Z*
dtype0
r
Adam/Ug/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ug/v_1
k
Adam/Ug/v_1/Read/ReadVariableOpReadVariableOpAdam/Ug/v_1*
_output_shapes

:Z*
dtype0
r
Adam/Ur/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ur/v_1
k
Adam/Ur/v_1/Read/ReadVariableOpReadVariableOpAdam/Ur/v_1*
_output_shapes

:Z*
dtype0
r
Adam/Uh/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uh/v_1
k
Adam/Uh/v_1/Read/ReadVariableOpReadVariableOpAdam/Uh/v_1*
_output_shapes

:Z*
dtype0
r
Adam/Wz/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wz/v_1
k
Adam/Wz/v_1/Read/ReadVariableOpReadVariableOpAdam/Wz/v_1*
_output_shapes

:ZZ*
dtype0
r
Adam/Wg/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wg/v_1
k
Adam/Wg/v_1/Read/ReadVariableOpReadVariableOpAdam/Wg/v_1*
_output_shapes

:ZZ*
dtype0
r
Adam/Wr/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wr/v_1
k
Adam/Wr/v_1/Read/ReadVariableOpReadVariableOpAdam/Wr/v_1*
_output_shapes

:ZZ*
dtype0
r
Adam/Wh/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wh/v_1
k
Adam/Wh/v_1/Read/ReadVariableOpReadVariableOpAdam/Wh/v_1*
_output_shapes

:ZZ*
dtype0
n
Adam/bz/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bz/v_1
g
Adam/bz/v_1/Read/ReadVariableOpReadVariableOpAdam/bz/v_1*
_output_shapes
:Z*
dtype0
n
Adam/bg/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bg/v_1
g
Adam/bg/v_1/Read/ReadVariableOpReadVariableOpAdam/bg/v_1*
_output_shapes
:Z*
dtype0
n
Adam/br/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/br/v_1
g
Adam/br/v_1/Read/ReadVariableOpReadVariableOpAdam/br/v_1*
_output_shapes
:Z*
dtype0
n
Adam/bh/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bh/v_1
g
Adam/bh/v_1/Read/ReadVariableOpReadVariableOpAdam/bh/v_1*
_output_shapes
:Z*
dtype0
r
Adam/Uz/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uz/v_2
k
Adam/Uz/v_2/Read/ReadVariableOpReadVariableOpAdam/Uz/v_2*
_output_shapes

:Z*
dtype0
r
Adam/Ug/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ug/v_2
k
Adam/Ug/v_2/Read/ReadVariableOpReadVariableOpAdam/Ug/v_2*
_output_shapes

:Z*
dtype0
r
Adam/Ur/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Ur/v_2
k
Adam/Ur/v_2/Read/ReadVariableOpReadVariableOpAdam/Ur/v_2*
_output_shapes

:Z*
dtype0
r
Adam/Uh/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameAdam/Uh/v_2
k
Adam/Uh/v_2/Read/ReadVariableOpReadVariableOpAdam/Uh/v_2*
_output_shapes

:Z*
dtype0
r
Adam/Wz/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wz/v_2
k
Adam/Wz/v_2/Read/ReadVariableOpReadVariableOpAdam/Wz/v_2*
_output_shapes

:ZZ*
dtype0
r
Adam/Wg/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wg/v_2
k
Adam/Wg/v_2/Read/ReadVariableOpReadVariableOpAdam/Wg/v_2*
_output_shapes

:ZZ*
dtype0
r
Adam/Wr/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wr/v_2
k
Adam/Wr/v_2/Read/ReadVariableOpReadVariableOpAdam/Wr/v_2*
_output_shapes

:ZZ*
dtype0
r
Adam/Wh/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*
shared_nameAdam/Wh/v_2
k
Adam/Wh/v_2/Read/ReadVariableOpReadVariableOpAdam/Wh/v_2*
_output_shapes

:ZZ*
dtype0
n
Adam/bz/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bz/v_2
g
Adam/bz/v_2/Read/ReadVariableOpReadVariableOpAdam/bz/v_2*
_output_shapes
:Z*
dtype0
n
Adam/bg/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bg/v_2
g
Adam/bg/v_2/Read/ReadVariableOpReadVariableOpAdam/bg/v_2*
_output_shapes
:Z*
dtype0
n
Adam/br/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/br/v_2
g
Adam/br/v_2/Read/ReadVariableOpReadVariableOpAdam/br/v_2*
_output_shapes
:Z*
dtype0
n
Adam/bh/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_nameAdam/bh/v_2
g
Adam/bh/v_2/Read/ReadVariableOpReadVariableOpAdam/bh/v_2*
_output_shapes
:Z*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:Z*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???=
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *4E@
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *4E@
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *    
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *4E@
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ?B
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *???=
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  ??

NoOpNoOp
??
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-0
&layer-37
'layer-38
(layer_with_weights-1
(layer-39
)layer-40
*layer_with_weights-2
*layer-41
+layer-42
,layer_with_weights-3
,layer-43
-layer-44
.layer_with_weights-4
.layer-45
/layer-46
0layer-47
1	optimizer
2loss
3trainable_variables
4regularization_losses
5	variables
6	keras_api
7
signatures
 

8	keras_api

9	keras_api

:	keras_api

;	keras_api

<	keras_api

=	keras_api

>	keras_api

?	keras_api

@	keras_api

A	keras_api

B	keras_api

C	keras_api

D	keras_api

E	keras_api

F	keras_api

G	keras_api

H	keras_api

I	keras_api

J	keras_api

K	keras_api

L	keras_api

M	keras_api

N	keras_api

O	keras_api

P	keras_api

Q	keras_api

R	keras_api

S	keras_api

T	keras_api

U	keras_api

V	keras_api

W	keras_api

X	keras_api

Y	keras_api

Z	keras_api

[	keras_api
h

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api

b	keras_api
?
cUz
dUg
eUr
fUh
gWz
hWg
iWr
jWh
kbz
lbg
mbr
nbh
o	variables
ptrainable_variables
qregularization_losses
r	keras_api

s	keras_api
?
tUz
uUg
vUr
wUh
xWz
yWg
zWr
{Wh
|bz
}bg
~br
bh
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?	keras_api
?
?Uz
?Ug
?Ur
?Uh
?Wz
?Wg
?Wr
?Wh
?bz
?bg
?br
?bh
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?	keras_api

?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate\m?]m?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?\v?]v?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
\0
]1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
 
?
\0
]1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?
?metrics
 ?layer_regularization_losses
?layers
3trainable_variables
?non_trainable_variables
4regularization_losses
5	variables
?layer_metrics
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
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

\0
]1
 
?
?metrics
 ?layer_regularization_losses
?layers
^	variables
?non_trainable_variables
_trainable_variables
?layer_metrics
`regularization_losses
 
JH
VARIABLE_VALUEUz2layer_with_weights-1/Uz/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEUg2layer_with_weights-1/Ug/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEUr2layer_with_weights-1/Ur/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEUh2layer_with_weights-1/Uh/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEWz2layer_with_weights-1/Wz/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEWg2layer_with_weights-1/Wg/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEWr2layer_with_weights-1/Wr/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEWh2layer_with_weights-1/Wh/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEbz2layer_with_weights-1/bz/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEbg2layer_with_weights-1/bg/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEbr2layer_with_weights-1/br/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEbh2layer_with_weights-1/bh/.ATTRIBUTES/VARIABLE_VALUE
V
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
V
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11
 
?
?metrics
 ?layer_regularization_losses
?layers
o	variables
?non_trainable_variables
ptrainable_variables
?layer_metrics
qregularization_losses
 
LJ
VARIABLE_VALUEUz_12layer_with_weights-2/Uz/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEUg_12layer_with_weights-2/Ug/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEUr_12layer_with_weights-2/Ur/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEUh_12layer_with_weights-2/Uh/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWz_12layer_with_weights-2/Wz/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWg_12layer_with_weights-2/Wg/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWr_12layer_with_weights-2/Wr/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWh_12layer_with_weights-2/Wh/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbz_12layer_with_weights-2/bz/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbg_12layer_with_weights-2/bg/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbr_12layer_with_weights-2/br/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbh_12layer_with_weights-2/bh/.ATTRIBUTES/VARIABLE_VALUE
V
t0
u1
v2
w3
x4
y5
z6
{7
|8
}9
~10
11
V
t0
u1
v2
w3
x4
y5
z6
{7
|8
}9
~10
11
 
?
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
 
LJ
VARIABLE_VALUEUz_22layer_with_weights-3/Uz/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEUg_22layer_with_weights-3/Ug/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEUr_22layer_with_weights-3/Ur/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEUh_22layer_with_weights-3/Uh/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWz_22layer_with_weights-3/Wz/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWg_22layer_with_weights-3/Wg/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWr_22layer_with_weights-3/Wr/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEWh_22layer_with_weights-3/Wh/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbz_22layer_with_weights-3/bz/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbg_22layer_with_weights-3/bg/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbr_22layer_with_weights-3/br/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbh_22layer_with_weights-3/bh/.ATTRIBUTES/VARIABLE_VALUE
b
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
b
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
 
?
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
 
 
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
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
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
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
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Uz/mNlayer_with_weights-1/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Ug/mNlayer_with_weights-1/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Ur/mNlayer_with_weights-1/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Uh/mNlayer_with_weights-1/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wz/mNlayer_with_weights-1/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wg/mNlayer_with_weights-1/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wr/mNlayer_with_weights-1/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wh/mNlayer_with_weights-1/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/bz/mNlayer_with_weights-1/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/bg/mNlayer_with_weights-1/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/br/mNlayer_with_weights-1/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/bh/mNlayer_with_weights-1/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uz/m_1Nlayer_with_weights-2/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ug/m_1Nlayer_with_weights-2/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ur/m_1Nlayer_with_weights-2/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uh/m_1Nlayer_with_weights-2/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wz/m_1Nlayer_with_weights-2/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wg/m_1Nlayer_with_weights-2/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wr/m_1Nlayer_with_weights-2/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wh/m_1Nlayer_with_weights-2/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bz/m_1Nlayer_with_weights-2/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bg/m_1Nlayer_with_weights-2/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/br/m_1Nlayer_with_weights-2/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bh/m_1Nlayer_with_weights-2/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uz/m_2Nlayer_with_weights-3/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ug/m_2Nlayer_with_weights-3/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ur/m_2Nlayer_with_weights-3/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uh/m_2Nlayer_with_weights-3/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wz/m_2Nlayer_with_weights-3/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wg/m_2Nlayer_with_weights-3/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wr/m_2Nlayer_with_weights-3/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wh/m_2Nlayer_with_weights-3/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bz/m_2Nlayer_with_weights-3/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bg/m_2Nlayer_with_weights-3/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/br/m_2Nlayer_with_weights-3/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bh/m_2Nlayer_with_weights-3/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Uz/vNlayer_with_weights-1/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Ug/vNlayer_with_weights-1/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Ur/vNlayer_with_weights-1/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Uh/vNlayer_with_weights-1/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wz/vNlayer_with_weights-1/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wg/vNlayer_with_weights-1/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wr/vNlayer_with_weights-1/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/Wh/vNlayer_with_weights-1/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/bz/vNlayer_with_weights-1/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/bg/vNlayer_with_weights-1/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/br/vNlayer_with_weights-1/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE	Adam/bh/vNlayer_with_weights-1/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uz/v_1Nlayer_with_weights-2/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ug/v_1Nlayer_with_weights-2/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ur/v_1Nlayer_with_weights-2/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uh/v_1Nlayer_with_weights-2/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wz/v_1Nlayer_with_weights-2/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wg/v_1Nlayer_with_weights-2/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wr/v_1Nlayer_with_weights-2/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wh/v_1Nlayer_with_weights-2/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bz/v_1Nlayer_with_weights-2/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bg/v_1Nlayer_with_weights-2/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/br/v_1Nlayer_with_weights-2/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bh/v_1Nlayer_with_weights-2/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uz/v_2Nlayer_with_weights-3/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ug/v_2Nlayer_with_weights-3/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Ur/v_2Nlayer_with_weights-3/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Uh/v_2Nlayer_with_weights-3/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wz/v_2Nlayer_with_weights-3/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wg/v_2Nlayer_with_weights-3/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wr/v_2Nlayer_with_weights-3/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Wh/v_2Nlayer_with_weights-3/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bz/v_2Nlayer_with_weights-3/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bg/v_2Nlayer_with_weights-3/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/br/v_2Nlayer_with_weights-3/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bh/v_2Nlayer_with_weights-3/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ConstConst_1Const_2Const_3Const_4Const_5Const_6dense/kernel
dense/biasUzWzbzUgWgbgUrWrbrUhWhbhConst_7Uz_1Wz_1bz_1Ug_1Wg_1bg_1Ur_1Wr_1br_1Uh_1Wh_1bh_1Uz_2Wz_2bz_2Ug_2Wg_2bg_2Ur_2Wr_2br_2Uh_2Wh_2bh_2dense_1/kerneldense_1/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_102542
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpUz/Read/ReadVariableOpUg/Read/ReadVariableOpUr/Read/ReadVariableOpUh/Read/ReadVariableOpWz/Read/ReadVariableOpWg/Read/ReadVariableOpWr/Read/ReadVariableOpWh/Read/ReadVariableOpbz/Read/ReadVariableOpbg/Read/ReadVariableOpbr/Read/ReadVariableOpbh/Read/ReadVariableOpUz_1/Read/ReadVariableOpUg_1/Read/ReadVariableOpUr_1/Read/ReadVariableOpUh_1/Read/ReadVariableOpWz_1/Read/ReadVariableOpWg_1/Read/ReadVariableOpWr_1/Read/ReadVariableOpWh_1/Read/ReadVariableOpbz_1/Read/ReadVariableOpbg_1/Read/ReadVariableOpbr_1/Read/ReadVariableOpbh_1/Read/ReadVariableOpUz_2/Read/ReadVariableOpUg_2/Read/ReadVariableOpUr_2/Read/ReadVariableOpUh_2/Read/ReadVariableOpWz_2/Read/ReadVariableOpWg_2/Read/ReadVariableOpWr_2/Read/ReadVariableOpWh_2/Read/ReadVariableOpbz_2/Read/ReadVariableOpbg_2/Read/ReadVariableOpbr_2/Read/ReadVariableOpbh_2/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOpAdam/Uz/m/Read/ReadVariableOpAdam/Ug/m/Read/ReadVariableOpAdam/Ur/m/Read/ReadVariableOpAdam/Uh/m/Read/ReadVariableOpAdam/Wz/m/Read/ReadVariableOpAdam/Wg/m/Read/ReadVariableOpAdam/Wr/m/Read/ReadVariableOpAdam/Wh/m/Read/ReadVariableOpAdam/bz/m/Read/ReadVariableOpAdam/bg/m/Read/ReadVariableOpAdam/br/m/Read/ReadVariableOpAdam/bh/m/Read/ReadVariableOpAdam/Uz/m_1/Read/ReadVariableOpAdam/Ug/m_1/Read/ReadVariableOpAdam/Ur/m_1/Read/ReadVariableOpAdam/Uh/m_1/Read/ReadVariableOpAdam/Wz/m_1/Read/ReadVariableOpAdam/Wg/m_1/Read/ReadVariableOpAdam/Wr/m_1/Read/ReadVariableOpAdam/Wh/m_1/Read/ReadVariableOpAdam/bz/m_1/Read/ReadVariableOpAdam/bg/m_1/Read/ReadVariableOpAdam/br/m_1/Read/ReadVariableOpAdam/bh/m_1/Read/ReadVariableOpAdam/Uz/m_2/Read/ReadVariableOpAdam/Ug/m_2/Read/ReadVariableOpAdam/Ur/m_2/Read/ReadVariableOpAdam/Uh/m_2/Read/ReadVariableOpAdam/Wz/m_2/Read/ReadVariableOpAdam/Wg/m_2/Read/ReadVariableOpAdam/Wr/m_2/Read/ReadVariableOpAdam/Wh/m_2/Read/ReadVariableOpAdam/bz/m_2/Read/ReadVariableOpAdam/bg/m_2/Read/ReadVariableOpAdam/br/m_2/Read/ReadVariableOpAdam/bh/m_2/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpAdam/Uz/v/Read/ReadVariableOpAdam/Ug/v/Read/ReadVariableOpAdam/Ur/v/Read/ReadVariableOpAdam/Uh/v/Read/ReadVariableOpAdam/Wz/v/Read/ReadVariableOpAdam/Wg/v/Read/ReadVariableOpAdam/Wr/v/Read/ReadVariableOpAdam/Wh/v/Read/ReadVariableOpAdam/bz/v/Read/ReadVariableOpAdam/bg/v/Read/ReadVariableOpAdam/br/v/Read/ReadVariableOpAdam/bh/v/Read/ReadVariableOpAdam/Uz/v_1/Read/ReadVariableOpAdam/Ug/v_1/Read/ReadVariableOpAdam/Ur/v_1/Read/ReadVariableOpAdam/Uh/v_1/Read/ReadVariableOpAdam/Wz/v_1/Read/ReadVariableOpAdam/Wg/v_1/Read/ReadVariableOpAdam/Wr/v_1/Read/ReadVariableOpAdam/Wh/v_1/Read/ReadVariableOpAdam/bz/v_1/Read/ReadVariableOpAdam/bg/v_1/Read/ReadVariableOpAdam/br/v_1/Read/ReadVariableOpAdam/bh/v_1/Read/ReadVariableOpAdam/Uz/v_2/Read/ReadVariableOpAdam/Ug/v_2/Read/ReadVariableOpAdam/Ur/v_2/Read/ReadVariableOpAdam/Uh/v_2/Read/ReadVariableOpAdam/Wz/v_2/Read/ReadVariableOpAdam/Wg/v_2/Read/ReadVariableOpAdam/Wr/v_2/Read/ReadVariableOpAdam/Wh/v_2/Read/ReadVariableOpAdam/bz/v_2/Read/ReadVariableOpAdam/bg/v_2/Read/ReadVariableOpAdam/br/v_2/Read/ReadVariableOpAdam/bh/v_2/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst_8*?
Tin?
?2	*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_103960
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasUzUgUrUhWzWgWrWhbzbgbrbhUz_1Ug_1Ur_1Uh_1Wz_1Wg_1Wr_1Wh_1bz_1bg_1br_1bh_1Uz_2Ug_2Ur_2Uh_2Wz_2Wg_2Wr_2Wh_2bz_2bg_2br_2bh_2dense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateAdam/dense/kernel/mAdam/dense/bias/m	Adam/Uz/m	Adam/Ug/m	Adam/Ur/m	Adam/Uh/m	Adam/Wz/m	Adam/Wg/m	Adam/Wr/m	Adam/Wh/m	Adam/bz/m	Adam/bg/m	Adam/br/m	Adam/bh/mAdam/Uz/m_1Adam/Ug/m_1Adam/Ur/m_1Adam/Uh/m_1Adam/Wz/m_1Adam/Wg/m_1Adam/Wr/m_1Adam/Wh/m_1Adam/bz/m_1Adam/bg/m_1Adam/br/m_1Adam/bh/m_1Adam/Uz/m_2Adam/Ug/m_2Adam/Ur/m_2Adam/Uh/m_2Adam/Wz/m_2Adam/Wg/m_2Adam/Wr/m_2Adam/Wh/m_2Adam/bz/m_2Adam/bg/m_2Adam/br/m_2Adam/bh/m_2Adam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense/kernel/vAdam/dense/bias/v	Adam/Uz/v	Adam/Ug/v	Adam/Ur/v	Adam/Uh/v	Adam/Wz/v	Adam/Wg/v	Adam/Wr/v	Adam/Wh/v	Adam/bz/v	Adam/bg/v	Adam/br/v	Adam/bh/vAdam/Uz/v_1Adam/Ug/v_1Adam/Ur/v_1Adam/Uh/v_1Adam/Wz/v_1Adam/Wg/v_1Adam/Wr/v_1Adam/Wh/v_1Adam/bz/v_1Adam/bg/v_1Adam/br/v_1Adam/bh/v_1Adam/Uz/v_2Adam/Ug/v_2Adam/Ur/v_2Adam/Uh/v_2Adam/Wz/v_2Adam/Wg/v_2Adam/Wr/v_2Adam/Wh/v_2Adam/bz/v_2Adam/bg/v_2Adam/br/v_2Adam/bh/v_2Adam/dense_1/kernel/vAdam/dense_1/bias/v*?
Tin?
?2~*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_104345͈
ԧ
?&
!__inference__wrapped_model_101233
input_1-
)dpde_model_tf___operators___add_6_addv2_x-
)dpde_model_tf___operators___add_2_addv2_x-
)dpde_model_tf___operators___add_1_addv2_x+
'dpde_model_tf___operators___add_addv2_x-
)dpde_model_tf___operators___add_3_addv2_x'
#dpde_model_tf_math_multiply_6_mul_x'
#dpde_model_tf_math_multiply_7_mul_x3
/dpde_model_dense_matmul_readvariableop_resource4
0dpde_model_dense_biasadd_readvariableop_resource;
7dpde_model_highway_layer_matmul_readvariableop_resource=
9dpde_model_highway_layer_matmul_1_readvariableop_resource:
6dpde_model_highway_layer_add_1_readvariableop_resource=
9dpde_model_highway_layer_matmul_2_readvariableop_resource=
9dpde_model_highway_layer_matmul_3_readvariableop_resource:
6dpde_model_highway_layer_add_3_readvariableop_resource=
9dpde_model_highway_layer_matmul_4_readvariableop_resource=
9dpde_model_highway_layer_matmul_5_readvariableop_resource:
6dpde_model_highway_layer_add_5_readvariableop_resource=
9dpde_model_highway_layer_matmul_6_readvariableop_resource=
9dpde_model_highway_layer_matmul_7_readvariableop_resource:
6dpde_model_highway_layer_add_7_readvariableop_resource-
)dpde_model_tf___operators___add_7_addv2_x=
9dpde_model_highway_layer_1_matmul_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_1_readvariableop_resource<
8dpde_model_highway_layer_1_add_1_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_2_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_3_readvariableop_resource<
8dpde_model_highway_layer_1_add_3_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_4_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_5_readvariableop_resource<
8dpde_model_highway_layer_1_add_5_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_6_readvariableop_resource?
;dpde_model_highway_layer_1_matmul_7_readvariableop_resource<
8dpde_model_highway_layer_1_add_7_readvariableop_resource=
9dpde_model_highway_layer_2_matmul_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_1_readvariableop_resource<
8dpde_model_highway_layer_2_add_1_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_2_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_3_readvariableop_resource<
8dpde_model_highway_layer_2_add_3_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_4_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_5_readvariableop_resource<
8dpde_model_highway_layer_2_add_5_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_6_readvariableop_resource?
;dpde_model_highway_layer_2_matmul_7_readvariableop_resource<
8dpde_model_highway_layer_2_add_7_readvariableop_resource5
1dpde_model_dense_1_matmul_readvariableop_resource6
2dpde_model_dense_1_biasadd_readvariableop_resource
identity??'dpde_model/dense/BiasAdd/ReadVariableOp?&dpde_model/dense/MatMul/ReadVariableOp?)dpde_model/dense_1/BiasAdd/ReadVariableOp?(dpde_model/dense_1/MatMul/ReadVariableOp?.dpde_model/highway_layer/MatMul/ReadVariableOp?0dpde_model/highway_layer/MatMul_1/ReadVariableOp?0dpde_model/highway_layer/MatMul_2/ReadVariableOp?0dpde_model/highway_layer/MatMul_3/ReadVariableOp?0dpde_model/highway_layer/MatMul_4/ReadVariableOp?0dpde_model/highway_layer/MatMul_5/ReadVariableOp?0dpde_model/highway_layer/MatMul_6/ReadVariableOp?0dpde_model/highway_layer/MatMul_7/ReadVariableOp?-dpde_model/highway_layer/add_1/ReadVariableOp?-dpde_model/highway_layer/add_3/ReadVariableOp?-dpde_model/highway_layer/add_5/ReadVariableOp?-dpde_model/highway_layer/add_7/ReadVariableOp?0dpde_model/highway_layer_1/MatMul/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_1/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_2/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_3/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_4/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_5/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_6/ReadVariableOp?2dpde_model/highway_layer_1/MatMul_7/ReadVariableOp?/dpde_model/highway_layer_1/add_1/ReadVariableOp?/dpde_model/highway_layer_1/add_3/ReadVariableOp?/dpde_model/highway_layer_1/add_5/ReadVariableOp?/dpde_model/highway_layer_1/add_7/ReadVariableOp?0dpde_model/highway_layer_2/MatMul/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_1/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_2/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_3/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_4/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_5/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_6/ReadVariableOp?2dpde_model/highway_layer_2/MatMul_7/ReadVariableOp?/dpde_model/highway_layer_2/add_1/ReadVariableOp?/dpde_model/highway_layer_2/add_3/ReadVariableOp?/dpde_model/highway_layer_2/add_5/ReadVariableOp?/dpde_model/highway_layer_2/add_7/ReadVariableOp?
9dpde_model/tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2;
9dpde_model/tf.__operators__.getitem_4/strided_slice/stack?
;dpde_model/tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2=
;dpde_model/tf.__operators__.getitem_4/strided_slice/stack_1?
;dpde_model/tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;dpde_model/tf.__operators__.getitem_4/strided_slice/stack_2?
3dpde_model/tf.__operators__.getitem_4/strided_sliceStridedSliceinput_1Bdpde_model/tf.__operators__.getitem_4/strided_slice/stack:output:0Ddpde_model/tf.__operators__.getitem_4/strided_slice/stack_1:output:0Ddpde_model/tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask25
3dpde_model/tf.__operators__.getitem_4/strided_slice?
9dpde_model/tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2;
9dpde_model/tf.__operators__.getitem_2/strided_slice/stack?
;dpde_model/tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2=
;dpde_model/tf.__operators__.getitem_2/strided_slice/stack_1?
;dpde_model/tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;dpde_model/tf.__operators__.getitem_2/strided_slice/stack_2?
3dpde_model/tf.__operators__.getitem_2/strided_sliceStridedSliceinput_1Bdpde_model/tf.__operators__.getitem_2/strided_slice/stack:output:0Ddpde_model/tf.__operators__.getitem_2/strided_slice/stack_1:output:0Ddpde_model/tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask25
3dpde_model/tf.__operators__.getitem_2/strided_slice?
9dpde_model/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2;
9dpde_model/tf.__operators__.getitem_1/strided_slice/stack?
;dpde_model/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2=
;dpde_model/tf.__operators__.getitem_1/strided_slice/stack_1?
;dpde_model/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;dpde_model/tf.__operators__.getitem_1/strided_slice/stack_2?
3dpde_model/tf.__operators__.getitem_1/strided_sliceStridedSliceinput_1Bdpde_model/tf.__operators__.getitem_1/strided_slice/stack:output:0Ddpde_model/tf.__operators__.getitem_1/strided_slice/stack_1:output:0Ddpde_model/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask25
3dpde_model/tf.__operators__.getitem_1/strided_slice?
7dpde_model/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7dpde_model/tf.__operators__.getitem/strided_slice/stack?
9dpde_model/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2;
9dpde_model/tf.__operators__.getitem/strided_slice/stack_1?
9dpde_model/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9dpde_model/tf.__operators__.getitem/strided_slice/stack_2?
1dpde_model/tf.__operators__.getitem/strided_sliceStridedSliceinput_1@dpde_model/tf.__operators__.getitem/strided_slice/stack:output:0Bdpde_model/tf.__operators__.getitem/strided_slice/stack_1:output:0Bdpde_model/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask23
1dpde_model/tf.__operators__.getitem/strided_slice?
#dpde_model/tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dpde_model/tf.math.subtract_4/Sub/y?
!dpde_model/tf.math.subtract_4/SubSub<dpde_model/tf.__operators__.getitem_4/strided_slice:output:0,dpde_model/tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.subtract_4/Sub?
9dpde_model/tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       2;
9dpde_model/tf.__operators__.getitem_3/strided_slice/stack?
;dpde_model/tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2=
;dpde_model/tf.__operators__.getitem_3/strided_slice/stack_1?
;dpde_model/tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;dpde_model/tf.__operators__.getitem_3/strided_slice/stack_2?
3dpde_model/tf.__operators__.getitem_3/strided_sliceStridedSliceinput_1Bdpde_model/tf.__operators__.getitem_3/strided_slice/stack:output:0Ddpde_model/tf.__operators__.getitem_3/strided_slice/stack_1:output:0Ddpde_model/tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask25
3dpde_model/tf.__operators__.getitem_3/strided_slice?
#dpde_model/tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dpde_model/tf.math.subtract_2/Sub/y?
!dpde_model/tf.math.subtract_2/SubSub<dpde_model/tf.__operators__.getitem_2/strided_slice:output:0,dpde_model/tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.subtract_2/Sub?
#dpde_model/tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dpde_model/tf.math.subtract_1/Sub/y?
!dpde_model/tf.math.subtract_1/SubSub<dpde_model/tf.__operators__.getitem_1/strided_slice:output:0,dpde_model/tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.subtract_1/Sub?
!dpde_model/tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dpde_model/tf.math.subtract/Sub/y?
dpde_model/tf.math.subtract/SubSub:dpde_model/tf.__operators__.getitem/strided_slice:output:0*dpde_model/tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2!
dpde_model/tf.math.subtract/Sub?
#dpde_model/tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2%
#dpde_model/tf.math.multiply_4/Mul/y?
!dpde_model/tf.math.multiply_4/MulMul%dpde_model/tf.math.subtract_4/Sub:z:0,dpde_model/tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_4/Mul?
#dpde_model/tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dpde_model/tf.math.subtract_3/Sub/y?
!dpde_model/tf.math.subtract_3/SubSub<dpde_model/tf.__operators__.getitem_3/strided_slice:output:0,dpde_model/tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.subtract_3/Sub?
#dpde_model/tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2%
#dpde_model/tf.math.multiply_2/Mul/y?
!dpde_model/tf.math.multiply_2/MulMul%dpde_model/tf.math.subtract_2/Sub:z:0,dpde_model/tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_2/Mul?
#dpde_model/tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2%
#dpde_model/tf.math.multiply_1/Mul/y?
!dpde_model/tf.math.multiply_1/MulMul%dpde_model/tf.math.subtract_1/Sub:z:0,dpde_model/tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_1/Mul?
!dpde_model/tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2#
!dpde_model/tf.math.multiply/Mul/y?
dpde_model/tf.math.multiply/MulMul#dpde_model/tf.math.subtract/Sub:z:0*dpde_model/tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2!
dpde_model/tf.math.multiply/Mul?
&dpde_model/tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&dpde_model/tf.math.truediv_5/truediv/y?
$dpde_model/tf.math.truediv_5/truedivRealDiv%dpde_model/tf.math.multiply_4/Mul:z:0/dpde_model/tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2&
$dpde_model/tf.math.truediv_5/truediv?
#dpde_model/tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2%
#dpde_model/tf.math.multiply_3/Mul/y?
!dpde_model/tf.math.multiply_3/MulMul%dpde_model/tf.math.subtract_3/Sub:z:0,dpde_model/tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_3/Mul?
&dpde_model/tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&dpde_model/tf.math.truediv_2/truediv/y?
$dpde_model/tf.math.truediv_2/truedivRealDiv%dpde_model/tf.math.multiply_2/Mul:z:0/dpde_model/tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2&
$dpde_model/tf.math.truediv_2/truediv?
&dpde_model/tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&dpde_model/tf.math.truediv_1/truediv/y?
$dpde_model/tf.math.truediv_1/truedivRealDiv%dpde_model/tf.math.multiply_1/Mul:z:0/dpde_model/tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2&
$dpde_model/tf.math.truediv_1/truediv?
$dpde_model/tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$dpde_model/tf.math.truediv/truediv/y?
"dpde_model/tf.math.truediv/truedivRealDiv#dpde_model/tf.math.multiply/Mul:z:0-dpde_model/tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2$
"dpde_model/tf.math.truediv/truediv?
'dpde_model/tf.__operators__.add_6/AddV2AddV2)dpde_model_tf___operators___add_6_addv2_x(dpde_model/tf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_6/AddV2?
&dpde_model/tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&dpde_model/tf.math.truediv_3/truediv/y?
$dpde_model/tf.math.truediv_3/truedivRealDiv%dpde_model/tf.math.multiply_3/Mul:z:0/dpde_model/tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2&
$dpde_model/tf.math.truediv_3/truediv?
'dpde_model/tf.__operators__.add_2/AddV2AddV2)dpde_model_tf___operators___add_2_addv2_x(dpde_model/tf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_2/AddV2?
'dpde_model/tf.__operators__.add_1/AddV2AddV2)dpde_model_tf___operators___add_1_addv2_x(dpde_model/tf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_1/AddV2?
dpde_model/tf.math.negative/NegNeg+dpde_model/tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2!
dpde_model/tf.math.negative/Neg?
%dpde_model/tf.__operators__.add/AddV2AddV2'dpde_model_tf___operators___add_addv2_x&dpde_model/tf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2'
%dpde_model/tf.__operators__.add/AddV2?
'dpde_model/tf.__operators__.add_3/AddV2AddV2)dpde_model_tf___operators___add_3_addv2_x(dpde_model/tf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_3/AddV2?
dpde_model/tf.math.exp/ExpExp+dpde_model/tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
dpde_model/tf.math.exp/Exp?
dpde_model/tf.math.exp_1/ExpExp+dpde_model/tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
dpde_model/tf.math.exp_1/Exp?
!dpde_model/tf.math.multiply_5/MulMul#dpde_model/tf.math.negative/Neg:y:0)dpde_model/tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_5/Mul?
'dpde_model/tf.__operators__.add_4/AddV2AddV2dpde_model/tf.math.exp/Exp:y:0 dpde_model/tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_4/AddV2?
dpde_model/tf.math.exp_2/ExpExp+dpde_model/tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
dpde_model/tf.math.exp_2/Exp?
dpde_model/tf.math.exp_3/ExpExp%dpde_model/tf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
dpde_model/tf.math.exp_3/Exp?
'dpde_model/tf.__operators__.add_5/AddV2AddV2+dpde_model/tf.__operators__.add_4/AddV2:z:0 dpde_model/tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_5/AddV2?
&dpde_model/tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2(
&dpde_model/tf.math.truediv_4/truediv/y?
$dpde_model/tf.math.truediv_4/truedivRealDiv+dpde_model/tf.__operators__.add_5/AddV2:z:0/dpde_model/tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2&
$dpde_model/tf.math.truediv_4/truediv?
!dpde_model/tf.math.multiply_6/MulMul#dpde_model_tf_math_multiply_6_mul_x dpde_model/tf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_6/Mul?
!dpde_model/tf.math.subtract_5/SubSub(dpde_model/tf.math.truediv_4/truediv:z:0%dpde_model/tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.subtract_5/Sub?
!dpde_model/tf.math.multiply_7/MulMul#dpde_model_tf_math_multiply_7_mul_x%dpde_model/tf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2#
!dpde_model/tf.math.multiply_7/Mul?
&dpde_model/dense/MatMul/ReadVariableOpReadVariableOp/dpde_model_dense_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02(
&dpde_model/dense/MatMul/ReadVariableOp?
dpde_model/dense/MatMulMatMulinput_1.dpde_model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/dense/MatMul?
'dpde_model/dense/BiasAdd/ReadVariableOpReadVariableOp0dpde_model_dense_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02)
'dpde_model/dense/BiasAdd/ReadVariableOp?
dpde_model/dense/BiasAddBiasAdd!dpde_model/dense/MatMul:product:0/dpde_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/dense/BiasAdd?
dpde_model/dense/TanhTanh!dpde_model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/dense/Tanh?
dpde_model/tf.math.exp_4/ExpExp%dpde_model/tf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
dpde_model/tf.math.exp_4/Exp?
.dpde_model/highway_layer/MatMul/ReadVariableOpReadVariableOp7dpde_model_highway_layer_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype020
.dpde_model/highway_layer/MatMul/ReadVariableOp?
dpde_model/highway_layer/MatMulMatMulinput_16dpde_model/highway_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2!
dpde_model/highway_layer/MatMul?
0dpde_model/highway_layer/MatMul_1/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype022
0dpde_model/highway_layer/MatMul_1/ReadVariableOp?
!dpde_model/highway_layer/MatMul_1MatMuldpde_model/dense/Tanh:y:08dpde_model/highway_layer/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_1?
dpde_model/highway_layer/addAddV2)dpde_model/highway_layer/MatMul:product:0+dpde_model/highway_layer/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/highway_layer/add?
-dpde_model/highway_layer/add_1/ReadVariableOpReadVariableOp6dpde_model_highway_layer_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02/
-dpde_model/highway_layer/add_1/ReadVariableOp?
dpde_model/highway_layer/add_1AddV2 dpde_model/highway_layer/add:z:05dpde_model/highway_layer/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_1?
dpde_model/highway_layer/TanhTanh"dpde_model/highway_layer/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/highway_layer/Tanh?
0dpde_model/highway_layer/MatMul_2/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype022
0dpde_model/highway_layer/MatMul_2/ReadVariableOp?
!dpde_model/highway_layer/MatMul_2MatMulinput_18dpde_model/highway_layer/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_2?
0dpde_model/highway_layer/MatMul_3/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype022
0dpde_model/highway_layer/MatMul_3/ReadVariableOp?
!dpde_model/highway_layer/MatMul_3MatMuldpde_model/dense/Tanh:y:08dpde_model/highway_layer/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_3?
dpde_model/highway_layer/add_2AddV2+dpde_model/highway_layer/MatMul_2:product:0+dpde_model/highway_layer/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_2?
-dpde_model/highway_layer/add_3/ReadVariableOpReadVariableOp6dpde_model_highway_layer_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02/
-dpde_model/highway_layer/add_3/ReadVariableOp?
dpde_model/highway_layer/add_3AddV2"dpde_model/highway_layer/add_2:z:05dpde_model/highway_layer/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_3?
dpde_model/highway_layer/Tanh_1Tanh"dpde_model/highway_layer/add_3:z:0*
T0*'
_output_shapes
:?????????Z2!
dpde_model/highway_layer/Tanh_1?
0dpde_model/highway_layer/MatMul_4/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype022
0dpde_model/highway_layer/MatMul_4/ReadVariableOp?
!dpde_model/highway_layer/MatMul_4MatMulinput_18dpde_model/highway_layer/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_4?
0dpde_model/highway_layer/MatMul_5/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype022
0dpde_model/highway_layer/MatMul_5/ReadVariableOp?
!dpde_model/highway_layer/MatMul_5MatMuldpde_model/dense/Tanh:y:08dpde_model/highway_layer/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_5?
dpde_model/highway_layer/add_4AddV2+dpde_model/highway_layer/MatMul_4:product:0+dpde_model/highway_layer/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_4?
-dpde_model/highway_layer/add_5/ReadVariableOpReadVariableOp6dpde_model_highway_layer_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02/
-dpde_model/highway_layer/add_5/ReadVariableOp?
dpde_model/highway_layer/add_5AddV2"dpde_model/highway_layer/add_4:z:05dpde_model/highway_layer/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_5?
dpde_model/highway_layer/Tanh_2Tanh"dpde_model/highway_layer/add_5:z:0*
T0*'
_output_shapes
:?????????Z2!
dpde_model/highway_layer/Tanh_2?
dpde_model/highway_layer/MulMuldpde_model/dense/Tanh:y:0#dpde_model/highway_layer/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/highway_layer/Mul?
0dpde_model/highway_layer/MatMul_6/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype022
0dpde_model/highway_layer/MatMul_6/ReadVariableOp?
!dpde_model/highway_layer/MatMul_6MatMulinput_18dpde_model/highway_layer/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_6?
0dpde_model/highway_layer/MatMul_7/ReadVariableOpReadVariableOp9dpde_model_highway_layer_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype022
0dpde_model/highway_layer/MatMul_7/ReadVariableOp?
!dpde_model/highway_layer/MatMul_7MatMul dpde_model/highway_layer/Mul:z:08dpde_model/highway_layer/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer/MatMul_7?
dpde_model/highway_layer/add_6AddV2+dpde_model/highway_layer/MatMul_6:product:0+dpde_model/highway_layer/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_6?
-dpde_model/highway_layer/add_7/ReadVariableOpReadVariableOp6dpde_model_highway_layer_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02/
-dpde_model/highway_layer/add_7/ReadVariableOp?
dpde_model/highway_layer/add_7AddV2"dpde_model/highway_layer/add_6:z:05dpde_model/highway_layer/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_7?
dpde_model/highway_layer/Tanh_3Tanh"dpde_model/highway_layer/add_7:z:0*
T0*'
_output_shapes
:?????????Z2!
dpde_model/highway_layer/Tanh_3?
(dpde_model/highway_layer/ones_like/ShapeShape#dpde_model/highway_layer/Tanh_1:y:0*
T0*
_output_shapes
:2*
(dpde_model/highway_layer/ones_like/Shape?
(dpde_model/highway_layer/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(dpde_model/highway_layer/ones_like/Const?
"dpde_model/highway_layer/ones_likeFill1dpde_model/highway_layer/ones_like/Shape:output:01dpde_model/highway_layer/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2$
"dpde_model/highway_layer/ones_like?
dpde_model/highway_layer/subSub+dpde_model/highway_layer/ones_like:output:0#dpde_model/highway_layer/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
dpde_model/highway_layer/sub?
dpde_model/highway_layer/Mul_1Mul dpde_model/highway_layer/sub:z:0#dpde_model/highway_layer/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/Mul_1?
dpde_model/highway_layer/Mul_2Mul!dpde_model/highway_layer/Tanh:y:0dpde_model/dense/Tanh:y:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/Mul_2?
dpde_model/highway_layer/add_8AddV2"dpde_model/highway_layer/Mul_1:z:0"dpde_model/highway_layer/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer/add_8?
'dpde_model/tf.__operators__.add_7/AddV2AddV2)dpde_model_tf___operators___add_7_addv2_x dpde_model/tf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_7/AddV2?
0dpde_model/highway_layer_1/MatMul/ReadVariableOpReadVariableOp9dpde_model_highway_layer_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype022
0dpde_model/highway_layer_1/MatMul/ReadVariableOp?
!dpde_model/highway_layer_1/MatMulMatMulinput_18dpde_model/highway_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_1/MatMul?
2dpde_model/highway_layer_1/MatMul_1/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_1/MatMul_1/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_1MatMul"dpde_model/highway_layer/add_8:z:0:dpde_model/highway_layer_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_1?
dpde_model/highway_layer_1/addAddV2+dpde_model/highway_layer_1/MatMul:product:0-dpde_model/highway_layer_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer_1/add?
/dpde_model/highway_layer_1/add_1/ReadVariableOpReadVariableOp8dpde_model_highway_layer_1_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_1/add_1/ReadVariableOp?
 dpde_model/highway_layer_1/add_1AddV2"dpde_model/highway_layer_1/add:z:07dpde_model/highway_layer_1/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_1?
dpde_model/highway_layer_1/TanhTanh$dpde_model/highway_layer_1/add_1:z:0*
T0*'
_output_shapes
:?????????Z2!
dpde_model/highway_layer_1/Tanh?
2dpde_model/highway_layer_1/MatMul_2/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype024
2dpde_model/highway_layer_1/MatMul_2/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_2MatMulinput_1:dpde_model/highway_layer_1/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_2?
2dpde_model/highway_layer_1/MatMul_3/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_1/MatMul_3/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_3MatMul"dpde_model/highway_layer/add_8:z:0:dpde_model/highway_layer_1/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_3?
 dpde_model/highway_layer_1/add_2AddV2-dpde_model/highway_layer_1/MatMul_2:product:0-dpde_model/highway_layer_1/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_2?
/dpde_model/highway_layer_1/add_3/ReadVariableOpReadVariableOp8dpde_model_highway_layer_1_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_1/add_3/ReadVariableOp?
 dpde_model/highway_layer_1/add_3AddV2$dpde_model/highway_layer_1/add_2:z:07dpde_model/highway_layer_1/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_3?
!dpde_model/highway_layer_1/Tanh_1Tanh$dpde_model/highway_layer_1/add_3:z:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_1/Tanh_1?
2dpde_model/highway_layer_1/MatMul_4/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype024
2dpde_model/highway_layer_1/MatMul_4/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_4MatMulinput_1:dpde_model/highway_layer_1/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_4?
2dpde_model/highway_layer_1/MatMul_5/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_1/MatMul_5/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_5MatMul"dpde_model/highway_layer/add_8:z:0:dpde_model/highway_layer_1/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_5?
 dpde_model/highway_layer_1/add_4AddV2-dpde_model/highway_layer_1/MatMul_4:product:0-dpde_model/highway_layer_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_4?
/dpde_model/highway_layer_1/add_5/ReadVariableOpReadVariableOp8dpde_model_highway_layer_1_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_1/add_5/ReadVariableOp?
 dpde_model/highway_layer_1/add_5AddV2$dpde_model/highway_layer_1/add_4:z:07dpde_model/highway_layer_1/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_5?
!dpde_model/highway_layer_1/Tanh_2Tanh$dpde_model/highway_layer_1/add_5:z:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_1/Tanh_2?
dpde_model/highway_layer_1/MulMul"dpde_model/highway_layer/add_8:z:0%dpde_model/highway_layer_1/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer_1/Mul?
2dpde_model/highway_layer_1/MatMul_6/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype024
2dpde_model/highway_layer_1/MatMul_6/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_6MatMulinput_1:dpde_model/highway_layer_1/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_6?
2dpde_model/highway_layer_1/MatMul_7/ReadVariableOpReadVariableOp;dpde_model_highway_layer_1_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_1/MatMul_7/ReadVariableOp?
#dpde_model/highway_layer_1/MatMul_7MatMul"dpde_model/highway_layer_1/Mul:z:0:dpde_model/highway_layer_1/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_1/MatMul_7?
 dpde_model/highway_layer_1/add_6AddV2-dpde_model/highway_layer_1/MatMul_6:product:0-dpde_model/highway_layer_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_6?
/dpde_model/highway_layer_1/add_7/ReadVariableOpReadVariableOp8dpde_model_highway_layer_1_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_1/add_7/ReadVariableOp?
 dpde_model/highway_layer_1/add_7AddV2$dpde_model/highway_layer_1/add_6:z:07dpde_model/highway_layer_1/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_7?
!dpde_model/highway_layer_1/Tanh_3Tanh$dpde_model/highway_layer_1/add_7:z:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_1/Tanh_3?
*dpde_model/highway_layer_1/ones_like/ShapeShape%dpde_model/highway_layer_1/Tanh_1:y:0*
T0*
_output_shapes
:2,
*dpde_model/highway_layer_1/ones_like/Shape?
*dpde_model/highway_layer_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*dpde_model/highway_layer_1/ones_like/Const?
$dpde_model/highway_layer_1/ones_likeFill3dpde_model/highway_layer_1/ones_like/Shape:output:03dpde_model/highway_layer_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2&
$dpde_model/highway_layer_1/ones_like?
dpde_model/highway_layer_1/subSub-dpde_model/highway_layer_1/ones_like:output:0%dpde_model/highway_layer_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer_1/sub?
 dpde_model/highway_layer_1/Mul_1Mul"dpde_model/highway_layer_1/sub:z:0%dpde_model/highway_layer_1/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/Mul_1?
 dpde_model/highway_layer_1/Mul_2Mul#dpde_model/highway_layer_1/Tanh:y:0"dpde_model/highway_layer/add_8:z:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/Mul_2?
 dpde_model/highway_layer_1/add_8AddV2$dpde_model/highway_layer_1/Mul_1:z:0$dpde_model/highway_layer_1/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_1/add_8?
dpde_model/tf.math.log/LogLog+dpde_model/tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
dpde_model/tf.math.log/Log?
0dpde_model/highway_layer_2/MatMul/ReadVariableOpReadVariableOp9dpde_model_highway_layer_2_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype022
0dpde_model/highway_layer_2/MatMul/ReadVariableOp?
!dpde_model/highway_layer_2/MatMulMatMulinput_18dpde_model/highway_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_2/MatMul?
2dpde_model/highway_layer_2/MatMul_1/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_2/MatMul_1/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_1MatMul$dpde_model/highway_layer_1/add_8:z:0:dpde_model/highway_layer_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_1?
dpde_model/highway_layer_2/addAddV2+dpde_model/highway_layer_2/MatMul:product:0-dpde_model/highway_layer_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer_2/add?
/dpde_model/highway_layer_2/add_1/ReadVariableOpReadVariableOp8dpde_model_highway_layer_2_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_2/add_1/ReadVariableOp?
 dpde_model/highway_layer_2/add_1AddV2"dpde_model/highway_layer_2/add:z:07dpde_model/highway_layer_2/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_1?
dpde_model/highway_layer_2/TanhTanh$dpde_model/highway_layer_2/add_1:z:0*
T0*'
_output_shapes
:?????????Z2!
dpde_model/highway_layer_2/Tanh?
2dpde_model/highway_layer_2/MatMul_2/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype024
2dpde_model/highway_layer_2/MatMul_2/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_2MatMulinput_1:dpde_model/highway_layer_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_2?
2dpde_model/highway_layer_2/MatMul_3/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_2/MatMul_3/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_3MatMul$dpde_model/highway_layer_1/add_8:z:0:dpde_model/highway_layer_2/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_3?
 dpde_model/highway_layer_2/add_2AddV2-dpde_model/highway_layer_2/MatMul_2:product:0-dpde_model/highway_layer_2/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_2?
/dpde_model/highway_layer_2/add_3/ReadVariableOpReadVariableOp8dpde_model_highway_layer_2_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_2/add_3/ReadVariableOp?
 dpde_model/highway_layer_2/add_3AddV2$dpde_model/highway_layer_2/add_2:z:07dpde_model/highway_layer_2/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_3?
!dpde_model/highway_layer_2/Tanh_1Tanh$dpde_model/highway_layer_2/add_3:z:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_2/Tanh_1?
2dpde_model/highway_layer_2/MatMul_4/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype024
2dpde_model/highway_layer_2/MatMul_4/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_4MatMulinput_1:dpde_model/highway_layer_2/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_4?
2dpde_model/highway_layer_2/MatMul_5/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_2/MatMul_5/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_5MatMul$dpde_model/highway_layer_1/add_8:z:0:dpde_model/highway_layer_2/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_5?
 dpde_model/highway_layer_2/add_4AddV2-dpde_model/highway_layer_2/MatMul_4:product:0-dpde_model/highway_layer_2/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_4?
/dpde_model/highway_layer_2/add_5/ReadVariableOpReadVariableOp8dpde_model_highway_layer_2_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_2/add_5/ReadVariableOp?
 dpde_model/highway_layer_2/add_5AddV2$dpde_model/highway_layer_2/add_4:z:07dpde_model/highway_layer_2/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_5?
!dpde_model/highway_layer_2/Tanh_2Tanh$dpde_model/highway_layer_2/add_5:z:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_2/Tanh_2?
dpde_model/highway_layer_2/MulMul$dpde_model/highway_layer_1/add_8:z:0%dpde_model/highway_layer_2/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer_2/Mul?
2dpde_model/highway_layer_2/MatMul_6/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype024
2dpde_model/highway_layer_2/MatMul_6/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_6MatMulinput_1:dpde_model/highway_layer_2/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_6?
2dpde_model/highway_layer_2/MatMul_7/ReadVariableOpReadVariableOp;dpde_model_highway_layer_2_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype024
2dpde_model/highway_layer_2/MatMul_7/ReadVariableOp?
#dpde_model/highway_layer_2/MatMul_7MatMul"dpde_model/highway_layer_2/Mul:z:0:dpde_model/highway_layer_2/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2%
#dpde_model/highway_layer_2/MatMul_7?
 dpde_model/highway_layer_2/add_6AddV2-dpde_model/highway_layer_2/MatMul_6:product:0-dpde_model/highway_layer_2/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_6?
/dpde_model/highway_layer_2/add_7/ReadVariableOpReadVariableOp8dpde_model_highway_layer_2_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype021
/dpde_model/highway_layer_2/add_7/ReadVariableOp?
 dpde_model/highway_layer_2/add_7AddV2$dpde_model/highway_layer_2/add_6:z:07dpde_model/highway_layer_2/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_7?
!dpde_model/highway_layer_2/Tanh_3Tanh$dpde_model/highway_layer_2/add_7:z:0*
T0*'
_output_shapes
:?????????Z2#
!dpde_model/highway_layer_2/Tanh_3?
*dpde_model/highway_layer_2/ones_like/ShapeShape%dpde_model/highway_layer_2/Tanh_1:y:0*
T0*
_output_shapes
:2,
*dpde_model/highway_layer_2/ones_like/Shape?
*dpde_model/highway_layer_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*dpde_model/highway_layer_2/ones_like/Const?
$dpde_model/highway_layer_2/ones_likeFill3dpde_model/highway_layer_2/ones_like/Shape:output:03dpde_model/highway_layer_2/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2&
$dpde_model/highway_layer_2/ones_like?
dpde_model/highway_layer_2/subSub-dpde_model/highway_layer_2/ones_like:output:0%dpde_model/highway_layer_2/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2 
dpde_model/highway_layer_2/sub?
 dpde_model/highway_layer_2/Mul_1Mul"dpde_model/highway_layer_2/sub:z:0%dpde_model/highway_layer_2/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/Mul_1?
 dpde_model/highway_layer_2/Mul_2Mul#dpde_model/highway_layer_2/Tanh:y:0$dpde_model/highway_layer_1/add_8:z:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/Mul_2?
 dpde_model/highway_layer_2/add_8AddV2$dpde_model/highway_layer_2/Mul_1:z:0$dpde_model/highway_layer_2/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2"
 dpde_model/highway_layer_2/add_8?
(dpde_model/dense_1/MatMul/ReadVariableOpReadVariableOp1dpde_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02*
(dpde_model/dense_1/MatMul/ReadVariableOp?
dpde_model/dense_1/MatMulMatMul$dpde_model/highway_layer_2/add_8:z:00dpde_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dpde_model/dense_1/MatMul?
)dpde_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp2dpde_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)dpde_model/dense_1/BiasAdd/ReadVariableOp?
dpde_model/dense_1/BiasAddBiasAdd#dpde_model/dense_1/MatMul:product:01dpde_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dpde_model/dense_1/BiasAdd?
&dpde_model/tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2(
&dpde_model/tf.math.truediv_6/truediv/y?
$dpde_model/tf.math.truediv_6/truedivRealDivdpde_model/tf.math.log/Log:y:0/dpde_model/tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2&
$dpde_model/tf.math.truediv_6/truediv?
'dpde_model/tf.__operators__.add_8/AddV2AddV2#dpde_model/dense_1/BiasAdd:output:0(dpde_model/tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2)
'dpde_model/tf.__operators__.add_8/AddV2?
IdentityIdentity+dpde_model/tf.__operators__.add_8/AddV2:z:0(^dpde_model/dense/BiasAdd/ReadVariableOp'^dpde_model/dense/MatMul/ReadVariableOp*^dpde_model/dense_1/BiasAdd/ReadVariableOp)^dpde_model/dense_1/MatMul/ReadVariableOp/^dpde_model/highway_layer/MatMul/ReadVariableOp1^dpde_model/highway_layer/MatMul_1/ReadVariableOp1^dpde_model/highway_layer/MatMul_2/ReadVariableOp1^dpde_model/highway_layer/MatMul_3/ReadVariableOp1^dpde_model/highway_layer/MatMul_4/ReadVariableOp1^dpde_model/highway_layer/MatMul_5/ReadVariableOp1^dpde_model/highway_layer/MatMul_6/ReadVariableOp1^dpde_model/highway_layer/MatMul_7/ReadVariableOp.^dpde_model/highway_layer/add_1/ReadVariableOp.^dpde_model/highway_layer/add_3/ReadVariableOp.^dpde_model/highway_layer/add_5/ReadVariableOp.^dpde_model/highway_layer/add_7/ReadVariableOp1^dpde_model/highway_layer_1/MatMul/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_1/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_2/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_3/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_4/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_5/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_6/ReadVariableOp3^dpde_model/highway_layer_1/MatMul_7/ReadVariableOp0^dpde_model/highway_layer_1/add_1/ReadVariableOp0^dpde_model/highway_layer_1/add_3/ReadVariableOp0^dpde_model/highway_layer_1/add_5/ReadVariableOp0^dpde_model/highway_layer_1/add_7/ReadVariableOp1^dpde_model/highway_layer_2/MatMul/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_1/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_2/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_3/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_4/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_5/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_6/ReadVariableOp3^dpde_model/highway_layer_2/MatMul_7/ReadVariableOp0^dpde_model/highway_layer_2/add_1/ReadVariableOp0^dpde_model/highway_layer_2/add_3/ReadVariableOp0^dpde_model/highway_layer_2/add_5/ReadVariableOp0^dpde_model/highway_layer_2/add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2R
'dpde_model/dense/BiasAdd/ReadVariableOp'dpde_model/dense/BiasAdd/ReadVariableOp2P
&dpde_model/dense/MatMul/ReadVariableOp&dpde_model/dense/MatMul/ReadVariableOp2V
)dpde_model/dense_1/BiasAdd/ReadVariableOp)dpde_model/dense_1/BiasAdd/ReadVariableOp2T
(dpde_model/dense_1/MatMul/ReadVariableOp(dpde_model/dense_1/MatMul/ReadVariableOp2`
.dpde_model/highway_layer/MatMul/ReadVariableOp.dpde_model/highway_layer/MatMul/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_1/ReadVariableOp0dpde_model/highway_layer/MatMul_1/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_2/ReadVariableOp0dpde_model/highway_layer/MatMul_2/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_3/ReadVariableOp0dpde_model/highway_layer/MatMul_3/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_4/ReadVariableOp0dpde_model/highway_layer/MatMul_4/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_5/ReadVariableOp0dpde_model/highway_layer/MatMul_5/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_6/ReadVariableOp0dpde_model/highway_layer/MatMul_6/ReadVariableOp2d
0dpde_model/highway_layer/MatMul_7/ReadVariableOp0dpde_model/highway_layer/MatMul_7/ReadVariableOp2^
-dpde_model/highway_layer/add_1/ReadVariableOp-dpde_model/highway_layer/add_1/ReadVariableOp2^
-dpde_model/highway_layer/add_3/ReadVariableOp-dpde_model/highway_layer/add_3/ReadVariableOp2^
-dpde_model/highway_layer/add_5/ReadVariableOp-dpde_model/highway_layer/add_5/ReadVariableOp2^
-dpde_model/highway_layer/add_7/ReadVariableOp-dpde_model/highway_layer/add_7/ReadVariableOp2d
0dpde_model/highway_layer_1/MatMul/ReadVariableOp0dpde_model/highway_layer_1/MatMul/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_1/ReadVariableOp2dpde_model/highway_layer_1/MatMul_1/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_2/ReadVariableOp2dpde_model/highway_layer_1/MatMul_2/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_3/ReadVariableOp2dpde_model/highway_layer_1/MatMul_3/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_4/ReadVariableOp2dpde_model/highway_layer_1/MatMul_4/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_5/ReadVariableOp2dpde_model/highway_layer_1/MatMul_5/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_6/ReadVariableOp2dpde_model/highway_layer_1/MatMul_6/ReadVariableOp2h
2dpde_model/highway_layer_1/MatMul_7/ReadVariableOp2dpde_model/highway_layer_1/MatMul_7/ReadVariableOp2b
/dpde_model/highway_layer_1/add_1/ReadVariableOp/dpde_model/highway_layer_1/add_1/ReadVariableOp2b
/dpde_model/highway_layer_1/add_3/ReadVariableOp/dpde_model/highway_layer_1/add_3/ReadVariableOp2b
/dpde_model/highway_layer_1/add_5/ReadVariableOp/dpde_model/highway_layer_1/add_5/ReadVariableOp2b
/dpde_model/highway_layer_1/add_7/ReadVariableOp/dpde_model/highway_layer_1/add_7/ReadVariableOp2d
0dpde_model/highway_layer_2/MatMul/ReadVariableOp0dpde_model/highway_layer_2/MatMul/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_1/ReadVariableOp2dpde_model/highway_layer_2/MatMul_1/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_2/ReadVariableOp2dpde_model/highway_layer_2/MatMul_2/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_3/ReadVariableOp2dpde_model/highway_layer_2/MatMul_3/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_4/ReadVariableOp2dpde_model/highway_layer_2/MatMul_4/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_5/ReadVariableOp2dpde_model/highway_layer_2/MatMul_5/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_6/ReadVariableOp2dpde_model/highway_layer_2/MatMul_6/ReadVariableOp2h
2dpde_model/highway_layer_2/MatMul_7/ReadVariableOp2dpde_model/highway_layer_2/MatMul_7/ReadVariableOp2b
/dpde_model/highway_layer_2/add_1/ReadVariableOp/dpde_model/highway_layer_2/add_1/ReadVariableOp2b
/dpde_model/highway_layer_2/add_3/ReadVariableOp/dpde_model/highway_layer_2/add_3/ReadVariableOp2b
/dpde_model/highway_layer_2/add_5/ReadVariableOp/dpde_model/highway_layer_2/add_5/ReadVariableOp2b
/dpde_model/highway_layer_2/add_7/ReadVariableOp/dpde_model/highway_layer_2/add_7/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_dpde_model_layer_call_fn_102159
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
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

unknown_20

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dpde_model_layer_call_and_return_conditional_losses_1020602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
.__inference_highway_layer_layer_call_fn_103361$
 input_combined_original_variable!
input_combined_previous_layer
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall input_combined_original_variableinput_combined_previous_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_highway_layer_layer_call_and_return_conditional_losses_1013972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
'
_output_shapes
:?????????
:
_user_specified_name" input_combined/original_variable:fb
'
_output_shapes
:?????????Z
7
_user_specified_nameinput_combined/previous_layer
?7
?
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_101628
input_combined
input_combined_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource!
add_5_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource!
add_7_readvariableop_resource
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?add_1/ReadVariableOp?add_3/ReadVariableOp?add_5/ReadVariableOp?add_7/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOp{
MatMulMatMulinput_combinedMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMulinput_combined_1MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_1Q
TanhTanh	add_1:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMulinput_combinedMatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_2?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMulinput_combined_1MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_3q
add_2AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
add_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_3/ReadVariableOpr
add_3AddV2	add_2:z:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_1?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_4/ReadVariableOp?
MatMul_4MatMulinput_combinedMatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_5/ReadVariableOp?
MatMul_5MatMulinput_combined_1MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_5q
add_4AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
add_4?
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_2a
MulMulinput_combined_1
Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
Mul?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_6/ReadVariableOp?
MatMul_6MatMulinput_combinedMatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_7/ReadVariableOpz
MatMul_7MatMulMul:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_7q
add_6AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
add_6?
add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_7/ReadVariableOpr
add_7AddV2	add_6:z:0add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_7U
Tanh_3Tanh	add_7:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_3\
ones_like/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
	ones_likec
subSubones_like:output:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
sub\
Mul_1Mulsub:z:0
Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
Mul_1c
Mul_2MulTanh:y:0input_combined_1*
T0*'
_output_shapes
:?????????Z2
Mul_2_
add_8AddV2	Mul_1:z:0	Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
add_8?
IdentityIdentity	add_8:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^add_1/ReadVariableOp^add_3/ReadVariableOp^add_5/ReadVariableOp^add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp:W S
'
_output_shapes
:?????????
(
_user_specified_nameinput_combined:WS
'
_output_shapes
:?????????Z
(
_user_specified_nameinput_combined
?8
?
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_103505$
 input_combined_original_variable!
input_combined_previous_layer"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource!
add_5_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource!
add_7_readvariableop_resource
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?add_1/ReadVariableOp?add_3/ReadVariableOp?add_5/ReadVariableOp?add_7/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul input_combined_original_variableMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMulinput_combined_previous_layerMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_1Q
TanhTanh	add_1:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMul input_combined_original_variableMatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_2?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMulinput_combined_previous_layerMatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_3q
add_2AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
add_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_3/ReadVariableOpr
add_3AddV2	add_2:z:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_1?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_4/ReadVariableOp?
MatMul_4MatMul input_combined_original_variableMatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_5/ReadVariableOp?
MatMul_5MatMulinput_combined_previous_layerMatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_5q
add_4AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
add_4?
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_2n
MulMulinput_combined_previous_layer
Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
Mul?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_6/ReadVariableOp?
MatMul_6MatMul input_combined_original_variableMatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_7/ReadVariableOpz
MatMul_7MatMulMul:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_7q
add_6AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
add_6?
add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_7/ReadVariableOpr
add_7AddV2	add_6:z:0add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_7U
Tanh_3Tanh	add_7:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_3\
ones_like/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
	ones_likec
subSubones_like:output:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
sub\
Mul_1Mulsub:z:0
Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
Mul_1p
Mul_2MulTanh:y:0input_combined_previous_layer*
T0*'
_output_shapes
:?????????Z2
Mul_2_
add_8AddV2	Mul_1:z:0	Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
add_8?
IdentityIdentity	add_8:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^add_1/ReadVariableOp^add_3/ReadVariableOp^add_5/ReadVariableOp^add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp:i e
'
_output_shapes
:?????????
:
_user_specified_name" input_combined/original_variable:fb
'
_output_shapes
:?????????Z
7
_user_specified_nameinput_combined/previous_layer
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_103545

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
{
&__inference_dense_layer_call_fn_103274

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1013232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_dpde_model_layer_call_and_return_conditional_losses_101715
input_1"
tf___operators___add_6_addv2_x"
tf___operators___add_2_addv2_x"
tf___operators___add_1_addv2_x 
tf___operators___add_addv2_x"
tf___operators___add_3_addv2_x
tf_math_multiply_6_mul_x
tf_math_multiply_7_mul_x
dense_101334
dense_101336
highway_layer_101429
highway_layer_101431
highway_layer_101433
highway_layer_101435
highway_layer_101437
highway_layer_101439
highway_layer_101441
highway_layer_101443
highway_layer_101445
highway_layer_101447
highway_layer_101449
highway_layer_101451"
tf___operators___add_7_addv2_x
highway_layer_1_101545
highway_layer_1_101547
highway_layer_1_101549
highway_layer_1_101551
highway_layer_1_101553
highway_layer_1_101555
highway_layer_1_101557
highway_layer_1_101559
highway_layer_1_101561
highway_layer_1_101563
highway_layer_1_101565
highway_layer_1_101567
highway_layer_2_101660
highway_layer_2_101662
highway_layer_2_101664
highway_layer_2_101666
highway_layer_2_101668
highway_layer_2_101670
highway_layer_2_101672
highway_layer_2_101674
highway_layer_2_101676
highway_layer_2_101678
highway_layer_2_101680
highway_layer_2_101682
dense_1_101706
dense_1_101708
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?%highway_layer/StatefulPartitionedCall?'highway_layer_1/StatefulPartitionedCall?'highway_layer_2/StatefulPartitionedCall?
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_4/strided_slice/stack?
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_4/strided_slice/stack_1?
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_4/strided_slice/stack_2?
(tf.__operators__.getitem_4/strided_sliceStridedSliceinput_17tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_4/strided_slice?
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_2/strided_slice/stack?
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_2/strided_slice/stack_1?
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_2/strided_slice/stack_2?
(tf.__operators__.getitem_2/strided_sliceStridedSliceinput_17tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_2/strided_slice?
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_1/strided_slice/stack?
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_1/strided_slice/stack_1?
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_1/strided_slice/stack_2?
(tf.__operators__.getitem_1/strided_sliceStridedSliceinput_17tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_1/strided_slice?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinput_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&tf.__operators__.getitem/strided_slicey
tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_4/Sub/y?
tf.math.subtract_4/SubSub1tf.__operators__.getitem_4/strided_slice:output:0!tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_4/Sub?
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_3/strided_slice/stack?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_3/strided_slice/stack_1?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_3/strided_slice/stack_2?
(tf.__operators__.getitem_3/strided_sliceStridedSliceinput_17tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_3/strided_slicey
tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_2/Sub/y?
tf.math.subtract_2/SubSub1tf.__operators__.getitem_2/strided_slice:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_2/Suby
tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_1/Sub/y?
tf.math.subtract_1/SubSub1tf.__operators__.getitem_1/strided_slice:output:0!tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_1/Subu
tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract/Sub/y?
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Suby
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMultf.math.subtract_4/Sub:z:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_4/Muly
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_3/Sub/y?
tf.math.subtract_3/SubSub1tf.__operators__.getitem_3/strided_slice:output:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_3/Suby
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMultf.math.subtract_2/Sub:z:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_2/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMultf.math.subtract_1/Sub:z:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMultf.math.subtract/Sub:z:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply/Mul
tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_5/truediv/y?
tf.math.truediv_5/truedivRealDivtf.math.multiply_4/Mul:z:0$tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_5/truedivy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMultf.math.subtract_3/Sub:z:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_3/Mul
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_2/truediv/y?
tf.math.truediv_2/truedivRealDivtf.math.multiply_2/Mul:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_2/truediv
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_1/truediv/y?
tf.math.truediv_1/truedivRealDivtf.math.multiply_1/Mul:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_1/truediv{
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv/truediv/y?
tf.math.truediv/truedivRealDivtf.math.multiply/Mul:z:0"tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv/truediv?
tf.__operators__.add_6/AddV2AddV2tf___operators___add_6_addv2_xtf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_6/AddV2
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_3/truediv/y?
tf.math.truediv_3/truedivRealDivtf.math.multiply_3/Mul:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_3/truediv?
tf.__operators__.add_2/AddV2AddV2tf___operators___add_2_addv2_xtf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_2/AddV2?
tf.__operators__.add_1/AddV2AddV2tf___operators___add_1_addv2_xtf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_1/AddV2?
tf.math.negative/NegNeg tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.negative/Neg?
tf.__operators__.add/AddV2AddV2tf___operators___add_addv2_xtf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
tf.__operators__.add_3/AddV2AddV2tf___operators___add_3_addv2_xtf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_3/AddV2}
tf.math.exp/ExpExp tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp/Exp?
tf.math.exp_1/ExpExp tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_1/Exp?
tf.math.multiply_5/MulMultf.math.negative/Neg:y:0tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_5/Mul?
tf.__operators__.add_4/AddV2AddV2tf.math.exp/Exp:y:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_4/AddV2?
tf.math.exp_2/ExpExp tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_2/Exp{
tf.math.exp_3/ExpExptf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_3/Exp?
tf.__operators__.add_5/AddV2AddV2 tf.__operators__.add_4/AddV2:z:0tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_5/AddV2
tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf.math.truediv_4/truediv/y?
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_5/AddV2:z:0$tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_4/truediv?
tf.math.multiply_6/MulMultf_math_multiply_6_mul_xtf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_6/Mul?
tf.math.subtract_5/SubSubtf.math.truediv_4/truediv:z:0tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_5/Sub?
tf.math.multiply_7/MulMultf_math_multiply_7_mul_xtf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_7/Mul?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_101334dense_101336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1013232
dense/StatefulPartitionedCall{
tf.math.exp_4/ExpExptf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_4/Exp?
%highway_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense/StatefulPartitionedCall:output:0highway_layer_101429highway_layer_101431highway_layer_101433highway_layer_101435highway_layer_101437highway_layer_101439highway_layer_101441highway_layer_101443highway_layer_101445highway_layer_101447highway_layer_101449highway_layer_101451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_highway_layer_layer_call_and_return_conditional_losses_1013972'
%highway_layer/StatefulPartitionedCall?
tf.__operators__.add_7/AddV2AddV2tf___operators___add_7_addv2_xtf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_7/AddV2?
'highway_layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_1.highway_layer/StatefulPartitionedCall:output:0highway_layer_1_101545highway_layer_1_101547highway_layer_1_101549highway_layer_1_101551highway_layer_1_101553highway_layer_1_101555highway_layer_1_101557highway_layer_1_101559highway_layer_1_101561highway_layer_1_101563highway_layer_1_101565highway_layer_1_101567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_1015132)
'highway_layer_1/StatefulPartitionedCall}
tf.math.log/LogLog tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.log/Log?
'highway_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput_10highway_layer_1/StatefulPartitionedCall:output:0highway_layer_2_101660highway_layer_2_101662highway_layer_2_101664highway_layer_2_101666highway_layer_2_101668highway_layer_2_101670highway_layer_2_101672highway_layer_2_101674highway_layer_2_101676highway_layer_2_101678highway_layer_2_101680highway_layer_2_101682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_1016282)
'highway_layer_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall0highway_layer_2/StatefulPartitionedCall:output:0dense_1_101706dense_1_101708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1016952!
dense_1/StatefulPartitionedCall
tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
tf.math.truediv_6/truediv/y?
tf.math.truediv_6/truedivRealDivtf.math.log/Log:y:0$tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_6/truediv?
tf.__operators__.add_8/AddV2AddV2(dense_1/StatefulPartitionedCall:output:0tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_8/AddV2?
IdentityIdentity tf.__operators__.add_8/AddV2:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^highway_layer/StatefulPartitionedCall(^highway_layer_1/StatefulPartitionedCall(^highway_layer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%highway_layer/StatefulPartitionedCall%highway_layer/StatefulPartitionedCall2R
'highway_layer_1/StatefulPartitionedCall'highway_layer_1/StatefulPartitionedCall2R
'highway_layer_2/StatefulPartitionedCall'highway_layer_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_dpde_model_layer_call_fn_102431
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
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

unknown_20

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dpde_model_layer_call_and_return_conditional_losses_1023322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_dpde_model_layer_call_and_return_conditional_losses_102332

inputs"
tf___operators___add_6_addv2_x"
tf___operators___add_2_addv2_x"
tf___operators___add_1_addv2_x 
tf___operators___add_addv2_x"
tf___operators___add_3_addv2_x
tf_math_multiply_6_mul_x
tf_math_multiply_7_mul_x
dense_102239
dense_102241
highway_layer_102245
highway_layer_102247
highway_layer_102249
highway_layer_102251
highway_layer_102253
highway_layer_102255
highway_layer_102257
highway_layer_102259
highway_layer_102261
highway_layer_102263
highway_layer_102265
highway_layer_102267"
tf___operators___add_7_addv2_x
highway_layer_1_102272
highway_layer_1_102274
highway_layer_1_102276
highway_layer_1_102278
highway_layer_1_102280
highway_layer_1_102282
highway_layer_1_102284
highway_layer_1_102286
highway_layer_1_102288
highway_layer_1_102290
highway_layer_1_102292
highway_layer_1_102294
highway_layer_2_102298
highway_layer_2_102300
highway_layer_2_102302
highway_layer_2_102304
highway_layer_2_102306
highway_layer_2_102308
highway_layer_2_102310
highway_layer_2_102312
highway_layer_2_102314
highway_layer_2_102316
highway_layer_2_102318
highway_layer_2_102320
dense_1_102323
dense_1_102325
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?%highway_layer/StatefulPartitionedCall?'highway_layer_1/StatefulPartitionedCall?'highway_layer_2/StatefulPartitionedCall?
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_4/strided_slice/stack?
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_4/strided_slice/stack_1?
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_4/strided_slice/stack_2?
(tf.__operators__.getitem_4/strided_sliceStridedSliceinputs7tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_4/strided_slice?
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_2/strided_slice/stack?
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_2/strided_slice/stack_1?
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_2/strided_slice/stack_2?
(tf.__operators__.getitem_2/strided_sliceStridedSliceinputs7tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_2/strided_slice?
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_1/strided_slice/stack?
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_1/strided_slice/stack_1?
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_1/strided_slice/stack_2?
(tf.__operators__.getitem_1/strided_sliceStridedSliceinputs7tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_1/strided_slice?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&tf.__operators__.getitem/strided_slicey
tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_4/Sub/y?
tf.math.subtract_4/SubSub1tf.__operators__.getitem_4/strided_slice:output:0!tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_4/Sub?
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_3/strided_slice/stack?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_3/strided_slice/stack_1?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_3/strided_slice/stack_2?
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_3/strided_slicey
tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_2/Sub/y?
tf.math.subtract_2/SubSub1tf.__operators__.getitem_2/strided_slice:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_2/Suby
tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_1/Sub/y?
tf.math.subtract_1/SubSub1tf.__operators__.getitem_1/strided_slice:output:0!tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_1/Subu
tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract/Sub/y?
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Suby
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMultf.math.subtract_4/Sub:z:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_4/Muly
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_3/Sub/y?
tf.math.subtract_3/SubSub1tf.__operators__.getitem_3/strided_slice:output:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_3/Suby
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMultf.math.subtract_2/Sub:z:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_2/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMultf.math.subtract_1/Sub:z:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMultf.math.subtract/Sub:z:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply/Mul
tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_5/truediv/y?
tf.math.truediv_5/truedivRealDivtf.math.multiply_4/Mul:z:0$tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_5/truedivy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMultf.math.subtract_3/Sub:z:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_3/Mul
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_2/truediv/y?
tf.math.truediv_2/truedivRealDivtf.math.multiply_2/Mul:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_2/truediv
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_1/truediv/y?
tf.math.truediv_1/truedivRealDivtf.math.multiply_1/Mul:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_1/truediv{
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv/truediv/y?
tf.math.truediv/truedivRealDivtf.math.multiply/Mul:z:0"tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv/truediv?
tf.__operators__.add_6/AddV2AddV2tf___operators___add_6_addv2_xtf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_6/AddV2
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_3/truediv/y?
tf.math.truediv_3/truedivRealDivtf.math.multiply_3/Mul:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_3/truediv?
tf.__operators__.add_2/AddV2AddV2tf___operators___add_2_addv2_xtf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_2/AddV2?
tf.__operators__.add_1/AddV2AddV2tf___operators___add_1_addv2_xtf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_1/AddV2?
tf.math.negative/NegNeg tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.negative/Neg?
tf.__operators__.add/AddV2AddV2tf___operators___add_addv2_xtf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
tf.__operators__.add_3/AddV2AddV2tf___operators___add_3_addv2_xtf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_3/AddV2}
tf.math.exp/ExpExp tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp/Exp?
tf.math.exp_1/ExpExp tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_1/Exp?
tf.math.multiply_5/MulMultf.math.negative/Neg:y:0tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_5/Mul?
tf.__operators__.add_4/AddV2AddV2tf.math.exp/Exp:y:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_4/AddV2?
tf.math.exp_2/ExpExp tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_2/Exp{
tf.math.exp_3/ExpExptf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_3/Exp?
tf.__operators__.add_5/AddV2AddV2 tf.__operators__.add_4/AddV2:z:0tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_5/AddV2
tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf.math.truediv_4/truediv/y?
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_5/AddV2:z:0$tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_4/truediv?
tf.math.multiply_6/MulMultf_math_multiply_6_mul_xtf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_6/Mul?
tf.math.subtract_5/SubSubtf.math.truediv_4/truediv:z:0tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_5/Sub?
tf.math.multiply_7/MulMultf_math_multiply_7_mul_xtf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_7/Mul?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_102239dense_102241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1013232
dense/StatefulPartitionedCall{
tf.math.exp_4/ExpExptf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_4/Exp?
%highway_layer/StatefulPartitionedCallStatefulPartitionedCallinputs&dense/StatefulPartitionedCall:output:0highway_layer_102245highway_layer_102247highway_layer_102249highway_layer_102251highway_layer_102253highway_layer_102255highway_layer_102257highway_layer_102259highway_layer_102261highway_layer_102263highway_layer_102265highway_layer_102267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_highway_layer_layer_call_and_return_conditional_losses_1013972'
%highway_layer/StatefulPartitionedCall?
tf.__operators__.add_7/AddV2AddV2tf___operators___add_7_addv2_xtf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_7/AddV2?
'highway_layer_1/StatefulPartitionedCallStatefulPartitionedCallinputs.highway_layer/StatefulPartitionedCall:output:0highway_layer_1_102272highway_layer_1_102274highway_layer_1_102276highway_layer_1_102278highway_layer_1_102280highway_layer_1_102282highway_layer_1_102284highway_layer_1_102286highway_layer_1_102288highway_layer_1_102290highway_layer_1_102292highway_layer_1_102294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_1015132)
'highway_layer_1/StatefulPartitionedCall}
tf.math.log/LogLog tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.log/Log?
'highway_layer_2/StatefulPartitionedCallStatefulPartitionedCallinputs0highway_layer_1/StatefulPartitionedCall:output:0highway_layer_2_102298highway_layer_2_102300highway_layer_2_102302highway_layer_2_102304highway_layer_2_102306highway_layer_2_102308highway_layer_2_102310highway_layer_2_102312highway_layer_2_102314highway_layer_2_102316highway_layer_2_102318highway_layer_2_102320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_1016282)
'highway_layer_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall0highway_layer_2/StatefulPartitionedCall:output:0dense_1_102323dense_1_102325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1016952!
dense_1/StatefulPartitionedCall
tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
tf.math.truediv_6/truediv/y?
tf.math.truediv_6/truedivRealDivtf.math.log/Log:y:0$tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_6/truediv?
tf.__operators__.add_8/AddV2AddV2(dense_1/StatefulPartitionedCall:output:0tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_8/AddV2?
IdentityIdentity tf.__operators__.add_8/AddV2:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^highway_layer/StatefulPartitionedCall(^highway_layer_1/StatefulPartitionedCall(^highway_layer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%highway_layer/StatefulPartitionedCall%highway_layer/StatefulPartitionedCall2R
'highway_layer_1/StatefulPartitionedCall'highway_layer_1/StatefulPartitionedCall2R
'highway_layer_2/StatefulPartitionedCall'highway_layer_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_102542
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
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

unknown_20

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1012332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_dpde_model_layer_call_and_return_conditional_losses_101886
input_1"
tf___operators___add_6_addv2_x"
tf___operators___add_2_addv2_x"
tf___operators___add_1_addv2_x 
tf___operators___add_addv2_x"
tf___operators___add_3_addv2_x
tf_math_multiply_6_mul_x
tf_math_multiply_7_mul_x
dense_101793
dense_101795
highway_layer_101799
highway_layer_101801
highway_layer_101803
highway_layer_101805
highway_layer_101807
highway_layer_101809
highway_layer_101811
highway_layer_101813
highway_layer_101815
highway_layer_101817
highway_layer_101819
highway_layer_101821"
tf___operators___add_7_addv2_x
highway_layer_1_101826
highway_layer_1_101828
highway_layer_1_101830
highway_layer_1_101832
highway_layer_1_101834
highway_layer_1_101836
highway_layer_1_101838
highway_layer_1_101840
highway_layer_1_101842
highway_layer_1_101844
highway_layer_1_101846
highway_layer_1_101848
highway_layer_2_101852
highway_layer_2_101854
highway_layer_2_101856
highway_layer_2_101858
highway_layer_2_101860
highway_layer_2_101862
highway_layer_2_101864
highway_layer_2_101866
highway_layer_2_101868
highway_layer_2_101870
highway_layer_2_101872
highway_layer_2_101874
dense_1_101877
dense_1_101879
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?%highway_layer/StatefulPartitionedCall?'highway_layer_1/StatefulPartitionedCall?'highway_layer_2/StatefulPartitionedCall?
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_4/strided_slice/stack?
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_4/strided_slice/stack_1?
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_4/strided_slice/stack_2?
(tf.__operators__.getitem_4/strided_sliceStridedSliceinput_17tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_4/strided_slice?
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_2/strided_slice/stack?
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_2/strided_slice/stack_1?
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_2/strided_slice/stack_2?
(tf.__operators__.getitem_2/strided_sliceStridedSliceinput_17tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_2/strided_slice?
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_1/strided_slice/stack?
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_1/strided_slice/stack_1?
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_1/strided_slice/stack_2?
(tf.__operators__.getitem_1/strided_sliceStridedSliceinput_17tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_1/strided_slice?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinput_15tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&tf.__operators__.getitem/strided_slicey
tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_4/Sub/y?
tf.math.subtract_4/SubSub1tf.__operators__.getitem_4/strided_slice:output:0!tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_4/Sub?
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_3/strided_slice/stack?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_3/strided_slice/stack_1?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_3/strided_slice/stack_2?
(tf.__operators__.getitem_3/strided_sliceStridedSliceinput_17tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_3/strided_slicey
tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_2/Sub/y?
tf.math.subtract_2/SubSub1tf.__operators__.getitem_2/strided_slice:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_2/Suby
tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_1/Sub/y?
tf.math.subtract_1/SubSub1tf.__operators__.getitem_1/strided_slice:output:0!tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_1/Subu
tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract/Sub/y?
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Suby
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMultf.math.subtract_4/Sub:z:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_4/Muly
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_3/Sub/y?
tf.math.subtract_3/SubSub1tf.__operators__.getitem_3/strided_slice:output:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_3/Suby
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMultf.math.subtract_2/Sub:z:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_2/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMultf.math.subtract_1/Sub:z:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMultf.math.subtract/Sub:z:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply/Mul
tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_5/truediv/y?
tf.math.truediv_5/truedivRealDivtf.math.multiply_4/Mul:z:0$tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_5/truedivy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMultf.math.subtract_3/Sub:z:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_3/Mul
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_2/truediv/y?
tf.math.truediv_2/truedivRealDivtf.math.multiply_2/Mul:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_2/truediv
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_1/truediv/y?
tf.math.truediv_1/truedivRealDivtf.math.multiply_1/Mul:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_1/truediv{
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv/truediv/y?
tf.math.truediv/truedivRealDivtf.math.multiply/Mul:z:0"tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv/truediv?
tf.__operators__.add_6/AddV2AddV2tf___operators___add_6_addv2_xtf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_6/AddV2
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_3/truediv/y?
tf.math.truediv_3/truedivRealDivtf.math.multiply_3/Mul:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_3/truediv?
tf.__operators__.add_2/AddV2AddV2tf___operators___add_2_addv2_xtf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_2/AddV2?
tf.__operators__.add_1/AddV2AddV2tf___operators___add_1_addv2_xtf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_1/AddV2?
tf.math.negative/NegNeg tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.negative/Neg?
tf.__operators__.add/AddV2AddV2tf___operators___add_addv2_xtf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
tf.__operators__.add_3/AddV2AddV2tf___operators___add_3_addv2_xtf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_3/AddV2}
tf.math.exp/ExpExp tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp/Exp?
tf.math.exp_1/ExpExp tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_1/Exp?
tf.math.multiply_5/MulMultf.math.negative/Neg:y:0tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_5/Mul?
tf.__operators__.add_4/AddV2AddV2tf.math.exp/Exp:y:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_4/AddV2?
tf.math.exp_2/ExpExp tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_2/Exp{
tf.math.exp_3/ExpExptf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_3/Exp?
tf.__operators__.add_5/AddV2AddV2 tf.__operators__.add_4/AddV2:z:0tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_5/AddV2
tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf.math.truediv_4/truediv/y?
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_5/AddV2:z:0$tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_4/truediv?
tf.math.multiply_6/MulMultf_math_multiply_6_mul_xtf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_6/Mul?
tf.math.subtract_5/SubSubtf.math.truediv_4/truediv:z:0tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_5/Sub?
tf.math.multiply_7/MulMultf_math_multiply_7_mul_xtf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_7/Mul?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_101793dense_101795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1013232
dense/StatefulPartitionedCall{
tf.math.exp_4/ExpExptf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_4/Exp?
%highway_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense/StatefulPartitionedCall:output:0highway_layer_101799highway_layer_101801highway_layer_101803highway_layer_101805highway_layer_101807highway_layer_101809highway_layer_101811highway_layer_101813highway_layer_101815highway_layer_101817highway_layer_101819highway_layer_101821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_highway_layer_layer_call_and_return_conditional_losses_1013972'
%highway_layer/StatefulPartitionedCall?
tf.__operators__.add_7/AddV2AddV2tf___operators___add_7_addv2_xtf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_7/AddV2?
'highway_layer_1/StatefulPartitionedCallStatefulPartitionedCallinput_1.highway_layer/StatefulPartitionedCall:output:0highway_layer_1_101826highway_layer_1_101828highway_layer_1_101830highway_layer_1_101832highway_layer_1_101834highway_layer_1_101836highway_layer_1_101838highway_layer_1_101840highway_layer_1_101842highway_layer_1_101844highway_layer_1_101846highway_layer_1_101848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_1015132)
'highway_layer_1/StatefulPartitionedCall}
tf.math.log/LogLog tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.log/Log?
'highway_layer_2/StatefulPartitionedCallStatefulPartitionedCallinput_10highway_layer_1/StatefulPartitionedCall:output:0highway_layer_2_101852highway_layer_2_101854highway_layer_2_101856highway_layer_2_101858highway_layer_2_101860highway_layer_2_101862highway_layer_2_101864highway_layer_2_101866highway_layer_2_101868highway_layer_2_101870highway_layer_2_101872highway_layer_2_101874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_1016282)
'highway_layer_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall0highway_layer_2/StatefulPartitionedCall:output:0dense_1_101877dense_1_101879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1016952!
dense_1/StatefulPartitionedCall
tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
tf.math.truediv_6/truediv/y?
tf.math.truediv_6/truedivRealDivtf.math.log/Log:y:0$tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_6/truediv?
tf.__operators__.add_8/AddV2AddV2(dense_1/StatefulPartitionedCall:output:0tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_8/AddV2?
IdentityIdentity tf.__operators__.add_8/AddV2:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^highway_layer/StatefulPartitionedCall(^highway_layer_1/StatefulPartitionedCall(^highway_layer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%highway_layer/StatefulPartitionedCall%highway_layer/StatefulPartitionedCall2R
'highway_layer_1/StatefulPartitionedCall'highway_layer_1/StatefulPartitionedCall2R
'highway_layer_2/StatefulPartitionedCall'highway_layer_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
F__inference_dpde_model_layer_call_and_return_conditional_losses_102060

inputs"
tf___operators___add_6_addv2_x"
tf___operators___add_2_addv2_x"
tf___operators___add_1_addv2_x 
tf___operators___add_addv2_x"
tf___operators___add_3_addv2_x
tf_math_multiply_6_mul_x
tf_math_multiply_7_mul_x
dense_101967
dense_101969
highway_layer_101973
highway_layer_101975
highway_layer_101977
highway_layer_101979
highway_layer_101981
highway_layer_101983
highway_layer_101985
highway_layer_101987
highway_layer_101989
highway_layer_101991
highway_layer_101993
highway_layer_101995"
tf___operators___add_7_addv2_x
highway_layer_1_102000
highway_layer_1_102002
highway_layer_1_102004
highway_layer_1_102006
highway_layer_1_102008
highway_layer_1_102010
highway_layer_1_102012
highway_layer_1_102014
highway_layer_1_102016
highway_layer_1_102018
highway_layer_1_102020
highway_layer_1_102022
highway_layer_2_102026
highway_layer_2_102028
highway_layer_2_102030
highway_layer_2_102032
highway_layer_2_102034
highway_layer_2_102036
highway_layer_2_102038
highway_layer_2_102040
highway_layer_2_102042
highway_layer_2_102044
highway_layer_2_102046
highway_layer_2_102048
dense_1_102051
dense_1_102053
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?%highway_layer/StatefulPartitionedCall?'highway_layer_1/StatefulPartitionedCall?'highway_layer_2/StatefulPartitionedCall?
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_4/strided_slice/stack?
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_4/strided_slice/stack_1?
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_4/strided_slice/stack_2?
(tf.__operators__.getitem_4/strided_sliceStridedSliceinputs7tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_4/strided_slice?
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_2/strided_slice/stack?
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_2/strided_slice/stack_1?
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_2/strided_slice/stack_2?
(tf.__operators__.getitem_2/strided_sliceStridedSliceinputs7tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_2/strided_slice?
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_1/strided_slice/stack?
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_1/strided_slice/stack_1?
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_1/strided_slice/stack_2?
(tf.__operators__.getitem_1/strided_sliceStridedSliceinputs7tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_1/strided_slice?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&tf.__operators__.getitem/strided_slicey
tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_4/Sub/y?
tf.math.subtract_4/SubSub1tf.__operators__.getitem_4/strided_slice:output:0!tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_4/Sub?
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_3/strided_slice/stack?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_3/strided_slice/stack_1?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_3/strided_slice/stack_2?
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_3/strided_slicey
tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_2/Sub/y?
tf.math.subtract_2/SubSub1tf.__operators__.getitem_2/strided_slice:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_2/Suby
tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_1/Sub/y?
tf.math.subtract_1/SubSub1tf.__operators__.getitem_1/strided_slice:output:0!tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_1/Subu
tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract/Sub/y?
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Suby
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMultf.math.subtract_4/Sub:z:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_4/Muly
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_3/Sub/y?
tf.math.subtract_3/SubSub1tf.__operators__.getitem_3/strided_slice:output:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_3/Suby
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMultf.math.subtract_2/Sub:z:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_2/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMultf.math.subtract_1/Sub:z:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMultf.math.subtract/Sub:z:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply/Mul
tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_5/truediv/y?
tf.math.truediv_5/truedivRealDivtf.math.multiply_4/Mul:z:0$tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_5/truedivy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMultf.math.subtract_3/Sub:z:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_3/Mul
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_2/truediv/y?
tf.math.truediv_2/truedivRealDivtf.math.multiply_2/Mul:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_2/truediv
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_1/truediv/y?
tf.math.truediv_1/truedivRealDivtf.math.multiply_1/Mul:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_1/truediv{
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv/truediv/y?
tf.math.truediv/truedivRealDivtf.math.multiply/Mul:z:0"tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv/truediv?
tf.__operators__.add_6/AddV2AddV2tf___operators___add_6_addv2_xtf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_6/AddV2
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_3/truediv/y?
tf.math.truediv_3/truedivRealDivtf.math.multiply_3/Mul:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_3/truediv?
tf.__operators__.add_2/AddV2AddV2tf___operators___add_2_addv2_xtf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_2/AddV2?
tf.__operators__.add_1/AddV2AddV2tf___operators___add_1_addv2_xtf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_1/AddV2?
tf.math.negative/NegNeg tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.negative/Neg?
tf.__operators__.add/AddV2AddV2tf___operators___add_addv2_xtf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
tf.__operators__.add_3/AddV2AddV2tf___operators___add_3_addv2_xtf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_3/AddV2}
tf.math.exp/ExpExp tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp/Exp?
tf.math.exp_1/ExpExp tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_1/Exp?
tf.math.multiply_5/MulMultf.math.negative/Neg:y:0tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_5/Mul?
tf.__operators__.add_4/AddV2AddV2tf.math.exp/Exp:y:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_4/AddV2?
tf.math.exp_2/ExpExp tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_2/Exp{
tf.math.exp_3/ExpExptf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_3/Exp?
tf.__operators__.add_5/AddV2AddV2 tf.__operators__.add_4/AddV2:z:0tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_5/AddV2
tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf.math.truediv_4/truediv/y?
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_5/AddV2:z:0$tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_4/truediv?
tf.math.multiply_6/MulMultf_math_multiply_6_mul_xtf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_6/Mul?
tf.math.subtract_5/SubSubtf.math.truediv_4/truediv:z:0tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_5/Sub?
tf.math.multiply_7/MulMultf_math_multiply_7_mul_xtf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_7/Mul?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_101967dense_101969*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_1013232
dense/StatefulPartitionedCall{
tf.math.exp_4/ExpExptf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_4/Exp?
%highway_layer/StatefulPartitionedCallStatefulPartitionedCallinputs&dense/StatefulPartitionedCall:output:0highway_layer_101973highway_layer_101975highway_layer_101977highway_layer_101979highway_layer_101981highway_layer_101983highway_layer_101985highway_layer_101987highway_layer_101989highway_layer_101991highway_layer_101993highway_layer_101995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_highway_layer_layer_call_and_return_conditional_losses_1013972'
%highway_layer/StatefulPartitionedCall?
tf.__operators__.add_7/AddV2AddV2tf___operators___add_7_addv2_xtf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_7/AddV2?
'highway_layer_1/StatefulPartitionedCallStatefulPartitionedCallinputs.highway_layer/StatefulPartitionedCall:output:0highway_layer_1_102000highway_layer_1_102002highway_layer_1_102004highway_layer_1_102006highway_layer_1_102008highway_layer_1_102010highway_layer_1_102012highway_layer_1_102014highway_layer_1_102016highway_layer_1_102018highway_layer_1_102020highway_layer_1_102022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_1015132)
'highway_layer_1/StatefulPartitionedCall}
tf.math.log/LogLog tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.log/Log?
'highway_layer_2/StatefulPartitionedCallStatefulPartitionedCallinputs0highway_layer_1/StatefulPartitionedCall:output:0highway_layer_2_102026highway_layer_2_102028highway_layer_2_102030highway_layer_2_102032highway_layer_2_102034highway_layer_2_102036highway_layer_2_102038highway_layer_2_102040highway_layer_2_102042highway_layer_2_102044highway_layer_2_102046highway_layer_2_102048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_1016282)
'highway_layer_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall0highway_layer_2/StatefulPartitionedCall:output:0dense_1_102051dense_1_102053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1016952!
dense_1/StatefulPartitionedCall
tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
tf.math.truediv_6/truediv/y?
tf.math.truediv_6/truedivRealDivtf.math.log/Log:y:0$tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_6/truediv?
tf.__operators__.add_8/AddV2AddV2(dense_1/StatefulPartitionedCall:output:0tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_8/AddV2?
IdentityIdentity tf.__operators__.add_8/AddV2:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall&^highway_layer/StatefulPartitionedCall(^highway_layer_1/StatefulPartitionedCall(^highway_layer_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2N
%highway_layer/StatefulPartitionedCall%highway_layer/StatefulPartitionedCall2R
'highway_layer_1/StatefulPartitionedCall'highway_layer_1/StatefulPartitionedCall2R
'highway_layer_2/StatefulPartitionedCall'highway_layer_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?)
__inference__traced_save_103960
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop!
savev2_uz_read_readvariableop!
savev2_ug_read_readvariableop!
savev2_ur_read_readvariableop!
savev2_uh_read_readvariableop!
savev2_wz_read_readvariableop!
savev2_wg_read_readvariableop!
savev2_wr_read_readvariableop!
savev2_wh_read_readvariableop!
savev2_bz_read_readvariableop!
savev2_bg_read_readvariableop!
savev2_br_read_readvariableop!
savev2_bh_read_readvariableop#
savev2_uz_1_read_readvariableop#
savev2_ug_1_read_readvariableop#
savev2_ur_1_read_readvariableop#
savev2_uh_1_read_readvariableop#
savev2_wz_1_read_readvariableop#
savev2_wg_1_read_readvariableop#
savev2_wr_1_read_readvariableop#
savev2_wh_1_read_readvariableop#
savev2_bz_1_read_readvariableop#
savev2_bg_1_read_readvariableop#
savev2_br_1_read_readvariableop#
savev2_bh_1_read_readvariableop#
savev2_uz_2_read_readvariableop#
savev2_ug_2_read_readvariableop#
savev2_ur_2_read_readvariableop#
savev2_uh_2_read_readvariableop#
savev2_wz_2_read_readvariableop#
savev2_wg_2_read_readvariableop#
savev2_wr_2_read_readvariableop#
savev2_wh_2_read_readvariableop#
savev2_bz_2_read_readvariableop#
savev2_bg_2_read_readvariableop#
savev2_br_2_read_readvariableop#
savev2_bh_2_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop(
$savev2_adam_uz_m_read_readvariableop(
$savev2_adam_ug_m_read_readvariableop(
$savev2_adam_ur_m_read_readvariableop(
$savev2_adam_uh_m_read_readvariableop(
$savev2_adam_wz_m_read_readvariableop(
$savev2_adam_wg_m_read_readvariableop(
$savev2_adam_wr_m_read_readvariableop(
$savev2_adam_wh_m_read_readvariableop(
$savev2_adam_bz_m_read_readvariableop(
$savev2_adam_bg_m_read_readvariableop(
$savev2_adam_br_m_read_readvariableop(
$savev2_adam_bh_m_read_readvariableop*
&savev2_adam_uz_m_1_read_readvariableop*
&savev2_adam_ug_m_1_read_readvariableop*
&savev2_adam_ur_m_1_read_readvariableop*
&savev2_adam_uh_m_1_read_readvariableop*
&savev2_adam_wz_m_1_read_readvariableop*
&savev2_adam_wg_m_1_read_readvariableop*
&savev2_adam_wr_m_1_read_readvariableop*
&savev2_adam_wh_m_1_read_readvariableop*
&savev2_adam_bz_m_1_read_readvariableop*
&savev2_adam_bg_m_1_read_readvariableop*
&savev2_adam_br_m_1_read_readvariableop*
&savev2_adam_bh_m_1_read_readvariableop*
&savev2_adam_uz_m_2_read_readvariableop*
&savev2_adam_ug_m_2_read_readvariableop*
&savev2_adam_ur_m_2_read_readvariableop*
&savev2_adam_uh_m_2_read_readvariableop*
&savev2_adam_wz_m_2_read_readvariableop*
&savev2_adam_wg_m_2_read_readvariableop*
&savev2_adam_wr_m_2_read_readvariableop*
&savev2_adam_wh_m_2_read_readvariableop*
&savev2_adam_bz_m_2_read_readvariableop*
&savev2_adam_bg_m_2_read_readvariableop*
&savev2_adam_br_m_2_read_readvariableop*
&savev2_adam_bh_m_2_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop(
$savev2_adam_uz_v_read_readvariableop(
$savev2_adam_ug_v_read_readvariableop(
$savev2_adam_ur_v_read_readvariableop(
$savev2_adam_uh_v_read_readvariableop(
$savev2_adam_wz_v_read_readvariableop(
$savev2_adam_wg_v_read_readvariableop(
$savev2_adam_wr_v_read_readvariableop(
$savev2_adam_wh_v_read_readvariableop(
$savev2_adam_bz_v_read_readvariableop(
$savev2_adam_bg_v_read_readvariableop(
$savev2_adam_br_v_read_readvariableop(
$savev2_adam_bh_v_read_readvariableop*
&savev2_adam_uz_v_1_read_readvariableop*
&savev2_adam_ug_v_1_read_readvariableop*
&savev2_adam_ur_v_1_read_readvariableop*
&savev2_adam_uh_v_1_read_readvariableop*
&savev2_adam_wz_v_1_read_readvariableop*
&savev2_adam_wg_v_1_read_readvariableop*
&savev2_adam_wr_v_1_read_readvariableop*
&savev2_adam_wh_v_1_read_readvariableop*
&savev2_adam_bz_v_1_read_readvariableop*
&savev2_adam_bg_v_1_read_readvariableop*
&savev2_adam_br_v_1_read_readvariableop*
&savev2_adam_bh_v_1_read_readvariableop*
&savev2_adam_uz_v_2_read_readvariableop*
&savev2_adam_ug_v_2_read_readvariableop*
&savev2_adam_ur_v_2_read_readvariableop*
&savev2_adam_uh_v_2_read_readvariableop*
&savev2_adam_wz_v_2_read_readvariableop*
&savev2_adam_wg_v_2_read_readvariableop*
&savev2_adam_wr_v_2_read_readvariableop*
&savev2_adam_wh_v_2_read_readvariableop*
&savev2_adam_bz_v_2_read_readvariableop*
&savev2_adam_bg_v_2_read_readvariableop*
&savev2_adam_br_v_2_read_readvariableop*
&savev2_adam_bh_v_2_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const_8

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?E
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?D
value?DB?D~B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Uz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Ug/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Ur/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Uh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wr/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/bz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/bg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/br/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/bh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Uz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Ug/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Ur/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Uh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wr/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/bz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/bg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/br/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/bh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Uz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Ug/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Ur/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Uh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wr/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/bz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/bg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/br/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/bh/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?
value?B?~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_uz_read_readvariableopsavev2_ug_read_readvariableopsavev2_ur_read_readvariableopsavev2_uh_read_readvariableopsavev2_wz_read_readvariableopsavev2_wg_read_readvariableopsavev2_wr_read_readvariableopsavev2_wh_read_readvariableopsavev2_bz_read_readvariableopsavev2_bg_read_readvariableopsavev2_br_read_readvariableopsavev2_bh_read_readvariableopsavev2_uz_1_read_readvariableopsavev2_ug_1_read_readvariableopsavev2_ur_1_read_readvariableopsavev2_uh_1_read_readvariableopsavev2_wz_1_read_readvariableopsavev2_wg_1_read_readvariableopsavev2_wr_1_read_readvariableopsavev2_wh_1_read_readvariableopsavev2_bz_1_read_readvariableopsavev2_bg_1_read_readvariableopsavev2_br_1_read_readvariableopsavev2_bh_1_read_readvariableopsavev2_uz_2_read_readvariableopsavev2_ug_2_read_readvariableopsavev2_ur_2_read_readvariableopsavev2_uh_2_read_readvariableopsavev2_wz_2_read_readvariableopsavev2_wg_2_read_readvariableopsavev2_wr_2_read_readvariableopsavev2_wh_2_read_readvariableopsavev2_bz_2_read_readvariableopsavev2_bg_2_read_readvariableopsavev2_br_2_read_readvariableopsavev2_bh_2_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop$savev2_adam_uz_m_read_readvariableop$savev2_adam_ug_m_read_readvariableop$savev2_adam_ur_m_read_readvariableop$savev2_adam_uh_m_read_readvariableop$savev2_adam_wz_m_read_readvariableop$savev2_adam_wg_m_read_readvariableop$savev2_adam_wr_m_read_readvariableop$savev2_adam_wh_m_read_readvariableop$savev2_adam_bz_m_read_readvariableop$savev2_adam_bg_m_read_readvariableop$savev2_adam_br_m_read_readvariableop$savev2_adam_bh_m_read_readvariableop&savev2_adam_uz_m_1_read_readvariableop&savev2_adam_ug_m_1_read_readvariableop&savev2_adam_ur_m_1_read_readvariableop&savev2_adam_uh_m_1_read_readvariableop&savev2_adam_wz_m_1_read_readvariableop&savev2_adam_wg_m_1_read_readvariableop&savev2_adam_wr_m_1_read_readvariableop&savev2_adam_wh_m_1_read_readvariableop&savev2_adam_bz_m_1_read_readvariableop&savev2_adam_bg_m_1_read_readvariableop&savev2_adam_br_m_1_read_readvariableop&savev2_adam_bh_m_1_read_readvariableop&savev2_adam_uz_m_2_read_readvariableop&savev2_adam_ug_m_2_read_readvariableop&savev2_adam_ur_m_2_read_readvariableop&savev2_adam_uh_m_2_read_readvariableop&savev2_adam_wz_m_2_read_readvariableop&savev2_adam_wg_m_2_read_readvariableop&savev2_adam_wr_m_2_read_readvariableop&savev2_adam_wh_m_2_read_readvariableop&savev2_adam_bz_m_2_read_readvariableop&savev2_adam_bg_m_2_read_readvariableop&savev2_adam_br_m_2_read_readvariableop&savev2_adam_bh_m_2_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop$savev2_adam_uz_v_read_readvariableop$savev2_adam_ug_v_read_readvariableop$savev2_adam_ur_v_read_readvariableop$savev2_adam_uh_v_read_readvariableop$savev2_adam_wz_v_read_readvariableop$savev2_adam_wg_v_read_readvariableop$savev2_adam_wr_v_read_readvariableop$savev2_adam_wh_v_read_readvariableop$savev2_adam_bz_v_read_readvariableop$savev2_adam_bg_v_read_readvariableop$savev2_adam_br_v_read_readvariableop$savev2_adam_bh_v_read_readvariableop&savev2_adam_uz_v_1_read_readvariableop&savev2_adam_ug_v_1_read_readvariableop&savev2_adam_ur_v_1_read_readvariableop&savev2_adam_uh_v_1_read_readvariableop&savev2_adam_wz_v_1_read_readvariableop&savev2_adam_wg_v_1_read_readvariableop&savev2_adam_wr_v_1_read_readvariableop&savev2_adam_wh_v_1_read_readvariableop&savev2_adam_bz_v_1_read_readvariableop&savev2_adam_bg_v_1_read_readvariableop&savev2_adam_br_v_1_read_readvariableop&savev2_adam_bh_v_1_read_readvariableop&savev2_adam_uz_v_2_read_readvariableop&savev2_adam_ug_v_2_read_readvariableop&savev2_adam_ur_v_2_read_readvariableop&savev2_adam_uh_v_2_read_readvariableop&savev2_adam_wz_v_2_read_readvariableop&savev2_adam_wg_v_2_read_readvariableop&savev2_adam_wr_v_2_read_readvariableop&savev2_adam_wh_v_2_read_readvariableop&savev2_adam_bz_v_2_read_readvariableop&savev2_adam_bg_v_2_read_readvariableop&savev2_adam_br_v_2_read_readvariableop&savev2_adam_bh_v_2_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const_8"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2~	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:: : : : : :Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z::Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:Z:Z:Z:ZZ:ZZ:ZZ:ZZ:Z:Z:Z:Z:Z:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Z: 

_output_shapes
:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:ZZ:$ 

_output_shapes

:ZZ:$	 

_output_shapes

:ZZ:$
 

_output_shapes

:ZZ: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:ZZ:$ 

_output_shapes

:ZZ:$ 

_output_shapes

:ZZ:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z: 

_output_shapes
:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:Z:$ 

_output_shapes

:ZZ:$  

_output_shapes

:ZZ:$! 

_output_shapes

:ZZ:$" 

_output_shapes

:ZZ: #

_output_shapes
:Z: $

_output_shapes
:Z: %

_output_shapes
:Z: &

_output_shapes
:Z:$' 

_output_shapes

:Z: (

_output_shapes
::)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :$. 

_output_shapes

:Z: /

_output_shapes
:Z:$0 

_output_shapes

:Z:$1 

_output_shapes

:Z:$2 

_output_shapes

:Z:$3 

_output_shapes

:Z:$4 

_output_shapes

:ZZ:$5 

_output_shapes

:ZZ:$6 

_output_shapes

:ZZ:$7 

_output_shapes

:ZZ: 8

_output_shapes
:Z: 9

_output_shapes
:Z: :

_output_shapes
:Z: ;

_output_shapes
:Z:$< 

_output_shapes

:Z:$= 

_output_shapes

:Z:$> 

_output_shapes

:Z:$? 

_output_shapes

:Z:$@ 

_output_shapes

:ZZ:$A 

_output_shapes

:ZZ:$B 

_output_shapes

:ZZ:$C 

_output_shapes

:ZZ: D

_output_shapes
:Z: E

_output_shapes
:Z: F

_output_shapes
:Z: G

_output_shapes
:Z:$H 

_output_shapes

:Z:$I 

_output_shapes

:Z:$J 

_output_shapes

:Z:$K 

_output_shapes

:Z:$L 

_output_shapes

:ZZ:$M 

_output_shapes

:ZZ:$N 

_output_shapes

:ZZ:$O 

_output_shapes

:ZZ: P

_output_shapes
:Z: Q

_output_shapes
:Z: R

_output_shapes
:Z: S

_output_shapes
:Z:$T 

_output_shapes

:Z: U

_output_shapes
::$V 

_output_shapes

:Z: W

_output_shapes
:Z:$X 

_output_shapes

:Z:$Y 

_output_shapes

:Z:$Z 

_output_shapes

:Z:$[ 

_output_shapes

:Z:$\ 

_output_shapes

:ZZ:$] 

_output_shapes

:ZZ:$^ 

_output_shapes

:ZZ:$_ 

_output_shapes

:ZZ: `

_output_shapes
:Z: a

_output_shapes
:Z: b

_output_shapes
:Z: c

_output_shapes
:Z:$d 

_output_shapes

:Z:$e 

_output_shapes

:Z:$f 

_output_shapes

:Z:$g 

_output_shapes

:Z:$h 

_output_shapes

:ZZ:$i 

_output_shapes

:ZZ:$j 

_output_shapes

:ZZ:$k 

_output_shapes

:ZZ: l

_output_shapes
:Z: m

_output_shapes
:Z: n

_output_shapes
:Z: o

_output_shapes
:Z:$p 

_output_shapes

:Z:$q 

_output_shapes

:Z:$r 

_output_shapes

:Z:$s 

_output_shapes

:Z:$t 

_output_shapes

:ZZ:$u 

_output_shapes

:ZZ:$v 

_output_shapes

:ZZ:$w 

_output_shapes

:ZZ: x

_output_shapes
:Z: y

_output_shapes
:Z: z

_output_shapes
:Z: {

_output_shapes
:Z:$| 

_output_shapes

:Z: }

_output_shapes
::~

_output_shapes
: 
?7
?
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_101513
input_combined
input_combined_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource!
add_5_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource!
add_7_readvariableop_resource
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?add_1/ReadVariableOp?add_3/ReadVariableOp?add_5/ReadVariableOp?add_7/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOp{
MatMulMatMulinput_combinedMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMulinput_combined_1MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_1Q
TanhTanh	add_1:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMulinput_combinedMatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_2?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMulinput_combined_1MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_3q
add_2AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
add_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_3/ReadVariableOpr
add_3AddV2	add_2:z:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_1?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_4/ReadVariableOp?
MatMul_4MatMulinput_combinedMatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_5/ReadVariableOp?
MatMul_5MatMulinput_combined_1MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_5q
add_4AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
add_4?
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_2a
MulMulinput_combined_1
Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
Mul?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_6/ReadVariableOp?
MatMul_6MatMulinput_combinedMatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_7/ReadVariableOpz
MatMul_7MatMulMul:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_7q
add_6AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
add_6?
add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_7/ReadVariableOpr
add_7AddV2	add_6:z:0add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_7U
Tanh_3Tanh	add_7:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_3\
ones_like/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
	ones_likec
subSubones_like:output:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
sub\
Mul_1Mulsub:z:0
Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
Mul_1c
Mul_2MulTanh:y:0input_combined_1*
T0*'
_output_shapes
:?????????Z2
Mul_2_
add_8AddV2	Mul_1:z:0	Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
add_8?
IdentityIdentity	add_8:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^add_1/ReadVariableOp^add_3/ReadVariableOp^add_5/ReadVariableOp^add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp:W S
'
_output_shapes
:?????????
(
_user_specified_nameinput_combined:WS
'
_output_shapes
:?????????Z
(
_user_specified_nameinput_combined
?
}
(__inference_dense_1_layer_call_fn_103554

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_1016952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
??
?8
"__inference__traced_restore_104345
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias
assignvariableop_2_uz
assignvariableop_3_ug
assignvariableop_4_ur
assignvariableop_5_uh
assignvariableop_6_wz
assignvariableop_7_wg
assignvariableop_8_wr
assignvariableop_9_wh
assignvariableop_10_bz
assignvariableop_11_bg
assignvariableop_12_br
assignvariableop_13_bh
assignvariableop_14_uz_1
assignvariableop_15_ug_1
assignvariableop_16_ur_1
assignvariableop_17_uh_1
assignvariableop_18_wz_1
assignvariableop_19_wg_1
assignvariableop_20_wr_1
assignvariableop_21_wh_1
assignvariableop_22_bz_1
assignvariableop_23_bg_1
assignvariableop_24_br_1
assignvariableop_25_bh_1
assignvariableop_26_uz_2
assignvariableop_27_ug_2
assignvariableop_28_ur_2
assignvariableop_29_uh_2
assignvariableop_30_wz_2
assignvariableop_31_wg_2
assignvariableop_32_wr_2
assignvariableop_33_wh_2
assignvariableop_34_bz_2
assignvariableop_35_bg_2
assignvariableop_36_br_2
assignvariableop_37_bh_2&
"assignvariableop_38_dense_1_kernel$
 assignvariableop_39_dense_1_bias!
assignvariableop_40_adam_iter#
assignvariableop_41_adam_beta_1#
assignvariableop_42_adam_beta_2"
assignvariableop_43_adam_decay*
&assignvariableop_44_adam_learning_rate+
'assignvariableop_45_adam_dense_kernel_m)
%assignvariableop_46_adam_dense_bias_m!
assignvariableop_47_adam_uz_m!
assignvariableop_48_adam_ug_m!
assignvariableop_49_adam_ur_m!
assignvariableop_50_adam_uh_m!
assignvariableop_51_adam_wz_m!
assignvariableop_52_adam_wg_m!
assignvariableop_53_adam_wr_m!
assignvariableop_54_adam_wh_m!
assignvariableop_55_adam_bz_m!
assignvariableop_56_adam_bg_m!
assignvariableop_57_adam_br_m!
assignvariableop_58_adam_bh_m#
assignvariableop_59_adam_uz_m_1#
assignvariableop_60_adam_ug_m_1#
assignvariableop_61_adam_ur_m_1#
assignvariableop_62_adam_uh_m_1#
assignvariableop_63_adam_wz_m_1#
assignvariableop_64_adam_wg_m_1#
assignvariableop_65_adam_wr_m_1#
assignvariableop_66_adam_wh_m_1#
assignvariableop_67_adam_bz_m_1#
assignvariableop_68_adam_bg_m_1#
assignvariableop_69_adam_br_m_1#
assignvariableop_70_adam_bh_m_1#
assignvariableop_71_adam_uz_m_2#
assignvariableop_72_adam_ug_m_2#
assignvariableop_73_adam_ur_m_2#
assignvariableop_74_adam_uh_m_2#
assignvariableop_75_adam_wz_m_2#
assignvariableop_76_adam_wg_m_2#
assignvariableop_77_adam_wr_m_2#
assignvariableop_78_adam_wh_m_2#
assignvariableop_79_adam_bz_m_2#
assignvariableop_80_adam_bg_m_2#
assignvariableop_81_adam_br_m_2#
assignvariableop_82_adam_bh_m_2-
)assignvariableop_83_adam_dense_1_kernel_m+
'assignvariableop_84_adam_dense_1_bias_m+
'assignvariableop_85_adam_dense_kernel_v)
%assignvariableop_86_adam_dense_bias_v!
assignvariableop_87_adam_uz_v!
assignvariableop_88_adam_ug_v!
assignvariableop_89_adam_ur_v!
assignvariableop_90_adam_uh_v!
assignvariableop_91_adam_wz_v!
assignvariableop_92_adam_wg_v!
assignvariableop_93_adam_wr_v!
assignvariableop_94_adam_wh_v!
assignvariableop_95_adam_bz_v!
assignvariableop_96_adam_bg_v!
assignvariableop_97_adam_br_v!
assignvariableop_98_adam_bh_v#
assignvariableop_99_adam_uz_v_1$
 assignvariableop_100_adam_ug_v_1$
 assignvariableop_101_adam_ur_v_1$
 assignvariableop_102_adam_uh_v_1$
 assignvariableop_103_adam_wz_v_1$
 assignvariableop_104_adam_wg_v_1$
 assignvariableop_105_adam_wr_v_1$
 assignvariableop_106_adam_wh_v_1$
 assignvariableop_107_adam_bz_v_1$
 assignvariableop_108_adam_bg_v_1$
 assignvariableop_109_adam_br_v_1$
 assignvariableop_110_adam_bh_v_1$
 assignvariableop_111_adam_uz_v_2$
 assignvariableop_112_adam_ug_v_2$
 assignvariableop_113_adam_ur_v_2$
 assignvariableop_114_adam_uh_v_2$
 assignvariableop_115_adam_wz_v_2$
 assignvariableop_116_adam_wg_v_2$
 assignvariableop_117_adam_wr_v_2$
 assignvariableop_118_adam_wh_v_2$
 assignvariableop_119_adam_bz_v_2$
 assignvariableop_120_adam_bg_v_2$
 assignvariableop_121_adam_br_v_2$
 assignvariableop_122_adam_bh_v_2.
*assignvariableop_123_adam_dense_1_kernel_v,
(assignvariableop_124_adam_dense_1_bias_v
identity_126??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?E
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?D
value?DB?D~B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Uz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Ug/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Ur/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Uh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wr/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/Wh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/bz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/bg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/br/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-1/bh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Uz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Ug/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Ur/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Uh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wr/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/Wh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/bz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/bg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/br/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-2/bh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Uz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Ug/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Ur/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Uh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wr/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/Wh/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/bz/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/bg/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/br/.ATTRIBUTES/VARIABLE_VALUEB2layer_with_weights-3/bh/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ug/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ur/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wr/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bz/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bg/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/br/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bh/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-1/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-2/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ug/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Ur/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Uh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wr/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/Wh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bz/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bg/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/br/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBNlayer_with_weights-3/bh/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*?
value?B?~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2~	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_uzIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_ugIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_urIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_uhIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_wzIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_wgIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_wrIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_whIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_bzIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_bgIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_brIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_bhIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_uz_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_ug_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_ur_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_uh_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_wz_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_wg_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_wr_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_wh_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_bz_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_bg_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_br_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_bh_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_uz_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_ug_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_ur_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_uh_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_wz_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_wg_2Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_wr_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_wh_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_bz_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_bg_2Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_br_2Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_bh_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp"assignvariableop_38_dense_1_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp assignvariableop_39_dense_1_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_adam_iterIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_adam_beta_1Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_beta_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_decayIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_learning_rateIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_adam_uz_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_adam_ug_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_adam_ur_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_adam_uh_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_adam_wz_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_wg_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_wr_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_wh_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_bz_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_bg_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_br_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_bh_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_uz_m_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpassignvariableop_60_adam_ug_m_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpassignvariableop_61_adam_ur_m_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpassignvariableop_62_adam_uh_m_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpassignvariableop_63_adam_wz_m_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpassignvariableop_64_adam_wg_m_1Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_wr_m_1Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpassignvariableop_66_adam_wh_m_1Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpassignvariableop_67_adam_bz_m_1Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpassignvariableop_68_adam_bg_m_1Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpassignvariableop_69_adam_br_m_1Identity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpassignvariableop_70_adam_bh_m_1Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpassignvariableop_71_adam_uz_m_2Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpassignvariableop_72_adam_ug_m_2Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpassignvariableop_73_adam_ur_m_2Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpassignvariableop_74_adam_uh_m_2Identity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOpassignvariableop_75_adam_wz_m_2Identity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOpassignvariableop_76_adam_wg_m_2Identity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOpassignvariableop_77_adam_wr_m_2Identity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOpassignvariableop_78_adam_wh_m_2Identity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOpassignvariableop_79_adam_bz_m_2Identity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOpassignvariableop_80_adam_bg_m_2Identity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOpassignvariableop_81_adam_br_m_2Identity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpassignvariableop_82_adam_bh_m_2Identity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adam_dense_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adam_dense_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adam_dense_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp%assignvariableop_86_adam_dense_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpassignvariableop_87_adam_uz_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpassignvariableop_88_adam_ug_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpassignvariableop_89_adam_ur_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpassignvariableop_90_adam_uh_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOpassignvariableop_91_adam_wz_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOpassignvariableop_92_adam_wg_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOpassignvariableop_93_adam_wr_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOpassignvariableop_94_adam_wh_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpassignvariableop_95_adam_bz_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpassignvariableop_96_adam_bg_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpassignvariableop_97_adam_br_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOpassignvariableop_98_adam_bh_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOpassignvariableop_99_adam_uz_v_1Identity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp assignvariableop_100_adam_ug_v_1Identity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp assignvariableop_101_adam_ur_v_1Identity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp assignvariableop_102_adam_uh_v_1Identity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp assignvariableop_103_adam_wz_v_1Identity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp assignvariableop_104_adam_wg_v_1Identity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp assignvariableop_105_adam_wr_v_1Identity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp assignvariableop_106_adam_wh_v_1Identity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp assignvariableop_107_adam_bz_v_1Identity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp assignvariableop_108_adam_bg_v_1Identity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp assignvariableop_109_adam_br_v_1Identity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp assignvariableop_110_adam_bh_v_1Identity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp assignvariableop_111_adam_uz_v_2Identity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp assignvariableop_112_adam_ug_v_2Identity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp assignvariableop_113_adam_ur_v_2Identity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp assignvariableop_114_adam_uh_v_2Identity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp assignvariableop_115_adam_wz_v_2Identity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp assignvariableop_116_adam_wg_v_2Identity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp assignvariableop_117_adam_wr_v_2Identity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp assignvariableop_118_adam_wh_v_2Identity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp assignvariableop_119_adam_bz_v_2Identity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp assignvariableop_120_adam_bg_v_2Identity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp assignvariableop_121_adam_br_v_2Identity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp assignvariableop_122_adam_bh_v_2Identity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp*assignvariableop_123_adam_dense_1_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp(assignvariableop_124_adam_dense_1_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_125Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_125?
Identity_126IdentityIdentity_125:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_126"%
identity_126Identity_126:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242*
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
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
F__inference_dpde_model_layer_call_and_return_conditional_losses_103052

inputs"
tf___operators___add_6_addv2_x"
tf___operators___add_2_addv2_x"
tf___operators___add_1_addv2_x 
tf___operators___add_addv2_x"
tf___operators___add_3_addv2_x
tf_math_multiply_6_mul_x
tf_math_multiply_7_mul_x(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource0
,highway_layer_matmul_readvariableop_resource2
.highway_layer_matmul_1_readvariableop_resource/
+highway_layer_add_1_readvariableop_resource2
.highway_layer_matmul_2_readvariableop_resource2
.highway_layer_matmul_3_readvariableop_resource/
+highway_layer_add_3_readvariableop_resource2
.highway_layer_matmul_4_readvariableop_resource2
.highway_layer_matmul_5_readvariableop_resource/
+highway_layer_add_5_readvariableop_resource2
.highway_layer_matmul_6_readvariableop_resource2
.highway_layer_matmul_7_readvariableop_resource/
+highway_layer_add_7_readvariableop_resource"
tf___operators___add_7_addv2_x2
.highway_layer_1_matmul_readvariableop_resource4
0highway_layer_1_matmul_1_readvariableop_resource1
-highway_layer_1_add_1_readvariableop_resource4
0highway_layer_1_matmul_2_readvariableop_resource4
0highway_layer_1_matmul_3_readvariableop_resource1
-highway_layer_1_add_3_readvariableop_resource4
0highway_layer_1_matmul_4_readvariableop_resource4
0highway_layer_1_matmul_5_readvariableop_resource1
-highway_layer_1_add_5_readvariableop_resource4
0highway_layer_1_matmul_6_readvariableop_resource4
0highway_layer_1_matmul_7_readvariableop_resource1
-highway_layer_1_add_7_readvariableop_resource2
.highway_layer_2_matmul_readvariableop_resource4
0highway_layer_2_matmul_1_readvariableop_resource1
-highway_layer_2_add_1_readvariableop_resource4
0highway_layer_2_matmul_2_readvariableop_resource4
0highway_layer_2_matmul_3_readvariableop_resource1
-highway_layer_2_add_3_readvariableop_resource4
0highway_layer_2_matmul_4_readvariableop_resource4
0highway_layer_2_matmul_5_readvariableop_resource1
-highway_layer_2_add_5_readvariableop_resource4
0highway_layer_2_matmul_6_readvariableop_resource4
0highway_layer_2_matmul_7_readvariableop_resource1
-highway_layer_2_add_7_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?#highway_layer/MatMul/ReadVariableOp?%highway_layer/MatMul_1/ReadVariableOp?%highway_layer/MatMul_2/ReadVariableOp?%highway_layer/MatMul_3/ReadVariableOp?%highway_layer/MatMul_4/ReadVariableOp?%highway_layer/MatMul_5/ReadVariableOp?%highway_layer/MatMul_6/ReadVariableOp?%highway_layer/MatMul_7/ReadVariableOp?"highway_layer/add_1/ReadVariableOp?"highway_layer/add_3/ReadVariableOp?"highway_layer/add_5/ReadVariableOp?"highway_layer/add_7/ReadVariableOp?%highway_layer_1/MatMul/ReadVariableOp?'highway_layer_1/MatMul_1/ReadVariableOp?'highway_layer_1/MatMul_2/ReadVariableOp?'highway_layer_1/MatMul_3/ReadVariableOp?'highway_layer_1/MatMul_4/ReadVariableOp?'highway_layer_1/MatMul_5/ReadVariableOp?'highway_layer_1/MatMul_6/ReadVariableOp?'highway_layer_1/MatMul_7/ReadVariableOp?$highway_layer_1/add_1/ReadVariableOp?$highway_layer_1/add_3/ReadVariableOp?$highway_layer_1/add_5/ReadVariableOp?$highway_layer_1/add_7/ReadVariableOp?%highway_layer_2/MatMul/ReadVariableOp?'highway_layer_2/MatMul_1/ReadVariableOp?'highway_layer_2/MatMul_2/ReadVariableOp?'highway_layer_2/MatMul_3/ReadVariableOp?'highway_layer_2/MatMul_4/ReadVariableOp?'highway_layer_2/MatMul_5/ReadVariableOp?'highway_layer_2/MatMul_6/ReadVariableOp?'highway_layer_2/MatMul_7/ReadVariableOp?$highway_layer_2/add_1/ReadVariableOp?$highway_layer_2/add_3/ReadVariableOp?$highway_layer_2/add_5/ReadVariableOp?$highway_layer_2/add_7/ReadVariableOp?
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_4/strided_slice/stack?
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_4/strided_slice/stack_1?
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_4/strided_slice/stack_2?
(tf.__operators__.getitem_4/strided_sliceStridedSliceinputs7tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_4/strided_slice?
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_2/strided_slice/stack?
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_2/strided_slice/stack_1?
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_2/strided_slice/stack_2?
(tf.__operators__.getitem_2/strided_sliceStridedSliceinputs7tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_2/strided_slice?
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_1/strided_slice/stack?
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_1/strided_slice/stack_1?
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_1/strided_slice/stack_2?
(tf.__operators__.getitem_1/strided_sliceStridedSliceinputs7tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_1/strided_slice?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&tf.__operators__.getitem/strided_slicey
tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_4/Sub/y?
tf.math.subtract_4/SubSub1tf.__operators__.getitem_4/strided_slice:output:0!tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_4/Sub?
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_3/strided_slice/stack?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_3/strided_slice/stack_1?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_3/strided_slice/stack_2?
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_3/strided_slicey
tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_2/Sub/y?
tf.math.subtract_2/SubSub1tf.__operators__.getitem_2/strided_slice:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_2/Suby
tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_1/Sub/y?
tf.math.subtract_1/SubSub1tf.__operators__.getitem_1/strided_slice:output:0!tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_1/Subu
tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract/Sub/y?
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Suby
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMultf.math.subtract_4/Sub:z:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_4/Muly
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_3/Sub/y?
tf.math.subtract_3/SubSub1tf.__operators__.getitem_3/strided_slice:output:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_3/Suby
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMultf.math.subtract_2/Sub:z:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_2/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMultf.math.subtract_1/Sub:z:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMultf.math.subtract/Sub:z:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply/Mul
tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_5/truediv/y?
tf.math.truediv_5/truedivRealDivtf.math.multiply_4/Mul:z:0$tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_5/truedivy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMultf.math.subtract_3/Sub:z:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_3/Mul
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_2/truediv/y?
tf.math.truediv_2/truedivRealDivtf.math.multiply_2/Mul:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_2/truediv
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_1/truediv/y?
tf.math.truediv_1/truedivRealDivtf.math.multiply_1/Mul:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_1/truediv{
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv/truediv/y?
tf.math.truediv/truedivRealDivtf.math.multiply/Mul:z:0"tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv/truediv?
tf.__operators__.add_6/AddV2AddV2tf___operators___add_6_addv2_xtf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_6/AddV2
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_3/truediv/y?
tf.math.truediv_3/truedivRealDivtf.math.multiply_3/Mul:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_3/truediv?
tf.__operators__.add_2/AddV2AddV2tf___operators___add_2_addv2_xtf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_2/AddV2?
tf.__operators__.add_1/AddV2AddV2tf___operators___add_1_addv2_xtf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_1/AddV2?
tf.math.negative/NegNeg tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.negative/Neg?
tf.__operators__.add/AddV2AddV2tf___operators___add_addv2_xtf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
tf.__operators__.add_3/AddV2AddV2tf___operators___add_3_addv2_xtf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_3/AddV2}
tf.math.exp/ExpExp tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp/Exp?
tf.math.exp_1/ExpExp tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_1/Exp?
tf.math.multiply_5/MulMultf.math.negative/Neg:y:0tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_5/Mul?
tf.__operators__.add_4/AddV2AddV2tf.math.exp/Exp:y:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_4/AddV2?
tf.math.exp_2/ExpExp tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_2/Exp{
tf.math.exp_3/ExpExptf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_3/Exp?
tf.__operators__.add_5/AddV2AddV2 tf.__operators__.add_4/AddV2:z:0tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_5/AddV2
tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf.math.truediv_4/truediv/y?
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_5/AddV2:z:0$tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_4/truediv?
tf.math.multiply_6/MulMultf_math_multiply_6_mul_xtf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_6/Mul?
tf.math.subtract_5/SubSubtf.math.truediv_4/truediv:z:0tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_5/Sub?
tf.math.multiply_7/MulMultf_math_multiply_7_mul_xtf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_7/Mul?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense/BiasAddj

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2

dense/Tanh{
tf.math.exp_4/ExpExptf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_4/Exp?
#highway_layer/MatMul/ReadVariableOpReadVariableOp,highway_layer_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02%
#highway_layer/MatMul/ReadVariableOp?
highway_layer/MatMulMatMulinputs+highway_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul?
%highway_layer/MatMul_1/ReadVariableOpReadVariableOp.highway_layer_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_1/ReadVariableOp?
highway_layer/MatMul_1MatMuldense/Tanh:y:0-highway_layer/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_1?
highway_layer/addAddV2highway_layer/MatMul:product:0 highway_layer/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add?
"highway_layer/add_1/ReadVariableOpReadVariableOp+highway_layer_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_1/ReadVariableOp?
highway_layer/add_1AddV2highway_layer/add:z:0*highway_layer/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_1{
highway_layer/TanhTanhhighway_layer/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh?
%highway_layer/MatMul_2/ReadVariableOpReadVariableOp.highway_layer_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer/MatMul_2/ReadVariableOp?
highway_layer/MatMul_2MatMulinputs-highway_layer/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_2?
%highway_layer/MatMul_3/ReadVariableOpReadVariableOp.highway_layer_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_3/ReadVariableOp?
highway_layer/MatMul_3MatMuldense/Tanh:y:0-highway_layer/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_3?
highway_layer/add_2AddV2 highway_layer/MatMul_2:product:0 highway_layer/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_2?
"highway_layer/add_3/ReadVariableOpReadVariableOp+highway_layer_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_3/ReadVariableOp?
highway_layer/add_3AddV2highway_layer/add_2:z:0*highway_layer/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_3
highway_layer/Tanh_1Tanhhighway_layer/add_3:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh_1?
%highway_layer/MatMul_4/ReadVariableOpReadVariableOp.highway_layer_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer/MatMul_4/ReadVariableOp?
highway_layer/MatMul_4MatMulinputs-highway_layer/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_4?
%highway_layer/MatMul_5/ReadVariableOpReadVariableOp.highway_layer_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_5/ReadVariableOp?
highway_layer/MatMul_5MatMuldense/Tanh:y:0-highway_layer/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_5?
highway_layer/add_4AddV2 highway_layer/MatMul_4:product:0 highway_layer/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_4?
"highway_layer/add_5/ReadVariableOpReadVariableOp+highway_layer_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_5/ReadVariableOp?
highway_layer/add_5AddV2highway_layer/add_4:z:0*highway_layer/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_5
highway_layer/Tanh_2Tanhhighway_layer/add_5:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh_2?
highway_layer/MulMuldense/Tanh:y:0highway_layer/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Mul?
%highway_layer/MatMul_6/ReadVariableOpReadVariableOp.highway_layer_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer/MatMul_6/ReadVariableOp?
highway_layer/MatMul_6MatMulinputs-highway_layer/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_6?
%highway_layer/MatMul_7/ReadVariableOpReadVariableOp.highway_layer_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_7/ReadVariableOp?
highway_layer/MatMul_7MatMulhighway_layer/Mul:z:0-highway_layer/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_7?
highway_layer/add_6AddV2 highway_layer/MatMul_6:product:0 highway_layer/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_6?
"highway_layer/add_7/ReadVariableOpReadVariableOp+highway_layer_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_7/ReadVariableOp?
highway_layer/add_7AddV2highway_layer/add_6:z:0*highway_layer/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_7
highway_layer/Tanh_3Tanhhighway_layer/add_7:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh_3?
highway_layer/ones_like/ShapeShapehighway_layer/Tanh_1:y:0*
T0*
_output_shapes
:2
highway_layer/ones_like/Shape?
highway_layer/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
highway_layer/ones_like/Const?
highway_layer/ones_likeFill&highway_layer/ones_like/Shape:output:0&highway_layer/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/ones_like?
highway_layer/subSub highway_layer/ones_like:output:0highway_layer/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/sub?
highway_layer/Mul_1Mulhighway_layer/sub:z:0highway_layer/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Mul_1?
highway_layer/Mul_2Mulhighway_layer/Tanh:y:0dense/Tanh:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Mul_2?
highway_layer/add_8AddV2highway_layer/Mul_1:z:0highway_layer/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_8?
tf.__operators__.add_7/AddV2AddV2tf___operators___add_7_addv2_xtf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_7/AddV2?
%highway_layer_1/MatMul/ReadVariableOpReadVariableOp.highway_layer_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer_1/MatMul/ReadVariableOp?
highway_layer_1/MatMulMatMulinputs-highway_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul?
'highway_layer_1/MatMul_1/ReadVariableOpReadVariableOp0highway_layer_1_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_1/ReadVariableOp?
highway_layer_1/MatMul_1MatMulhighway_layer/add_8:z:0/highway_layer_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_1?
highway_layer_1/addAddV2 highway_layer_1/MatMul:product:0"highway_layer_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add?
$highway_layer_1/add_1/ReadVariableOpReadVariableOp-highway_layer_1_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_1/ReadVariableOp?
highway_layer_1/add_1AddV2highway_layer_1/add:z:0,highway_layer_1/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_1?
highway_layer_1/TanhTanhhighway_layer_1/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh?
'highway_layer_1/MatMul_2/ReadVariableOpReadVariableOp0highway_layer_1_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_1/MatMul_2/ReadVariableOp?
highway_layer_1/MatMul_2MatMulinputs/highway_layer_1/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_2?
'highway_layer_1/MatMul_3/ReadVariableOpReadVariableOp0highway_layer_1_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_3/ReadVariableOp?
highway_layer_1/MatMul_3MatMulhighway_layer/add_8:z:0/highway_layer_1/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_3?
highway_layer_1/add_2AddV2"highway_layer_1/MatMul_2:product:0"highway_layer_1/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_2?
$highway_layer_1/add_3/ReadVariableOpReadVariableOp-highway_layer_1_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_3/ReadVariableOp?
highway_layer_1/add_3AddV2highway_layer_1/add_2:z:0,highway_layer_1/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_3?
highway_layer_1/Tanh_1Tanhhighway_layer_1/add_3:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh_1?
'highway_layer_1/MatMul_4/ReadVariableOpReadVariableOp0highway_layer_1_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_1/MatMul_4/ReadVariableOp?
highway_layer_1/MatMul_4MatMulinputs/highway_layer_1/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_4?
'highway_layer_1/MatMul_5/ReadVariableOpReadVariableOp0highway_layer_1_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_5/ReadVariableOp?
highway_layer_1/MatMul_5MatMulhighway_layer/add_8:z:0/highway_layer_1/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_5?
highway_layer_1/add_4AddV2"highway_layer_1/MatMul_4:product:0"highway_layer_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_4?
$highway_layer_1/add_5/ReadVariableOpReadVariableOp-highway_layer_1_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_5/ReadVariableOp?
highway_layer_1/add_5AddV2highway_layer_1/add_4:z:0,highway_layer_1/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_5?
highway_layer_1/Tanh_2Tanhhighway_layer_1/add_5:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh_2?
highway_layer_1/MulMulhighway_layer/add_8:z:0highway_layer_1/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Mul?
'highway_layer_1/MatMul_6/ReadVariableOpReadVariableOp0highway_layer_1_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_1/MatMul_6/ReadVariableOp?
highway_layer_1/MatMul_6MatMulinputs/highway_layer_1/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_6?
'highway_layer_1/MatMul_7/ReadVariableOpReadVariableOp0highway_layer_1_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_7/ReadVariableOp?
highway_layer_1/MatMul_7MatMulhighway_layer_1/Mul:z:0/highway_layer_1/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_7?
highway_layer_1/add_6AddV2"highway_layer_1/MatMul_6:product:0"highway_layer_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_6?
$highway_layer_1/add_7/ReadVariableOpReadVariableOp-highway_layer_1_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_7/ReadVariableOp?
highway_layer_1/add_7AddV2highway_layer_1/add_6:z:0,highway_layer_1/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_7?
highway_layer_1/Tanh_3Tanhhighway_layer_1/add_7:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh_3?
highway_layer_1/ones_like/ShapeShapehighway_layer_1/Tanh_1:y:0*
T0*
_output_shapes
:2!
highway_layer_1/ones_like/Shape?
highway_layer_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
highway_layer_1/ones_like/Const?
highway_layer_1/ones_likeFill(highway_layer_1/ones_like/Shape:output:0(highway_layer_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/ones_like?
highway_layer_1/subSub"highway_layer_1/ones_like:output:0highway_layer_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/sub?
highway_layer_1/Mul_1Mulhighway_layer_1/sub:z:0highway_layer_1/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Mul_1?
highway_layer_1/Mul_2Mulhighway_layer_1/Tanh:y:0highway_layer/add_8:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Mul_2?
highway_layer_1/add_8AddV2highway_layer_1/Mul_1:z:0highway_layer_1/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_8}
tf.math.log/LogLog tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.log/Log?
%highway_layer_2/MatMul/ReadVariableOpReadVariableOp.highway_layer_2_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer_2/MatMul/ReadVariableOp?
highway_layer_2/MatMulMatMulinputs-highway_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul?
'highway_layer_2/MatMul_1/ReadVariableOpReadVariableOp0highway_layer_2_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_1/ReadVariableOp?
highway_layer_2/MatMul_1MatMulhighway_layer_1/add_8:z:0/highway_layer_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_1?
highway_layer_2/addAddV2 highway_layer_2/MatMul:product:0"highway_layer_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add?
$highway_layer_2/add_1/ReadVariableOpReadVariableOp-highway_layer_2_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_1/ReadVariableOp?
highway_layer_2/add_1AddV2highway_layer_2/add:z:0,highway_layer_2/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_1?
highway_layer_2/TanhTanhhighway_layer_2/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh?
'highway_layer_2/MatMul_2/ReadVariableOpReadVariableOp0highway_layer_2_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_2/MatMul_2/ReadVariableOp?
highway_layer_2/MatMul_2MatMulinputs/highway_layer_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_2?
'highway_layer_2/MatMul_3/ReadVariableOpReadVariableOp0highway_layer_2_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_3/ReadVariableOp?
highway_layer_2/MatMul_3MatMulhighway_layer_1/add_8:z:0/highway_layer_2/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_3?
highway_layer_2/add_2AddV2"highway_layer_2/MatMul_2:product:0"highway_layer_2/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_2?
$highway_layer_2/add_3/ReadVariableOpReadVariableOp-highway_layer_2_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_3/ReadVariableOp?
highway_layer_2/add_3AddV2highway_layer_2/add_2:z:0,highway_layer_2/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_3?
highway_layer_2/Tanh_1Tanhhighway_layer_2/add_3:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh_1?
'highway_layer_2/MatMul_4/ReadVariableOpReadVariableOp0highway_layer_2_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_2/MatMul_4/ReadVariableOp?
highway_layer_2/MatMul_4MatMulinputs/highway_layer_2/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_4?
'highway_layer_2/MatMul_5/ReadVariableOpReadVariableOp0highway_layer_2_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_5/ReadVariableOp?
highway_layer_2/MatMul_5MatMulhighway_layer_1/add_8:z:0/highway_layer_2/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_5?
highway_layer_2/add_4AddV2"highway_layer_2/MatMul_4:product:0"highway_layer_2/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_4?
$highway_layer_2/add_5/ReadVariableOpReadVariableOp-highway_layer_2_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_5/ReadVariableOp?
highway_layer_2/add_5AddV2highway_layer_2/add_4:z:0,highway_layer_2/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_5?
highway_layer_2/Tanh_2Tanhhighway_layer_2/add_5:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh_2?
highway_layer_2/MulMulhighway_layer_1/add_8:z:0highway_layer_2/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Mul?
'highway_layer_2/MatMul_6/ReadVariableOpReadVariableOp0highway_layer_2_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_2/MatMul_6/ReadVariableOp?
highway_layer_2/MatMul_6MatMulinputs/highway_layer_2/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_6?
'highway_layer_2/MatMul_7/ReadVariableOpReadVariableOp0highway_layer_2_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_7/ReadVariableOp?
highway_layer_2/MatMul_7MatMulhighway_layer_2/Mul:z:0/highway_layer_2/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_7?
highway_layer_2/add_6AddV2"highway_layer_2/MatMul_6:product:0"highway_layer_2/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_6?
$highway_layer_2/add_7/ReadVariableOpReadVariableOp-highway_layer_2_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_7/ReadVariableOp?
highway_layer_2/add_7AddV2highway_layer_2/add_6:z:0,highway_layer_2/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_7?
highway_layer_2/Tanh_3Tanhhighway_layer_2/add_7:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh_3?
highway_layer_2/ones_like/ShapeShapehighway_layer_2/Tanh_1:y:0*
T0*
_output_shapes
:2!
highway_layer_2/ones_like/Shape?
highway_layer_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
highway_layer_2/ones_like/Const?
highway_layer_2/ones_likeFill(highway_layer_2/ones_like/Shape:output:0(highway_layer_2/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/ones_like?
highway_layer_2/subSub"highway_layer_2/ones_like:output:0highway_layer_2/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/sub?
highway_layer_2/Mul_1Mulhighway_layer_2/sub:z:0highway_layer_2/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Mul_1?
highway_layer_2/Mul_2Mulhighway_layer_2/Tanh:y:0highway_layer_1/add_8:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Mul_2?
highway_layer_2/add_8AddV2highway_layer_2/Mul_1:z:0highway_layer_2/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_8?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulhighway_layer_2/add_8:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd
tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
tf.math.truediv_6/truediv/y?
tf.math.truediv_6/truedivRealDivtf.math.log/Log:y:0$tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_6/truediv?
tf.__operators__.add_8/AddV2AddV2dense_1/BiasAdd:output:0tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_8/AddV2?
IdentityIdentity tf.__operators__.add_8/AddV2:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp$^highway_layer/MatMul/ReadVariableOp&^highway_layer/MatMul_1/ReadVariableOp&^highway_layer/MatMul_2/ReadVariableOp&^highway_layer/MatMul_3/ReadVariableOp&^highway_layer/MatMul_4/ReadVariableOp&^highway_layer/MatMul_5/ReadVariableOp&^highway_layer/MatMul_6/ReadVariableOp&^highway_layer/MatMul_7/ReadVariableOp#^highway_layer/add_1/ReadVariableOp#^highway_layer/add_3/ReadVariableOp#^highway_layer/add_5/ReadVariableOp#^highway_layer/add_7/ReadVariableOp&^highway_layer_1/MatMul/ReadVariableOp(^highway_layer_1/MatMul_1/ReadVariableOp(^highway_layer_1/MatMul_2/ReadVariableOp(^highway_layer_1/MatMul_3/ReadVariableOp(^highway_layer_1/MatMul_4/ReadVariableOp(^highway_layer_1/MatMul_5/ReadVariableOp(^highway_layer_1/MatMul_6/ReadVariableOp(^highway_layer_1/MatMul_7/ReadVariableOp%^highway_layer_1/add_1/ReadVariableOp%^highway_layer_1/add_3/ReadVariableOp%^highway_layer_1/add_5/ReadVariableOp%^highway_layer_1/add_7/ReadVariableOp&^highway_layer_2/MatMul/ReadVariableOp(^highway_layer_2/MatMul_1/ReadVariableOp(^highway_layer_2/MatMul_2/ReadVariableOp(^highway_layer_2/MatMul_3/ReadVariableOp(^highway_layer_2/MatMul_4/ReadVariableOp(^highway_layer_2/MatMul_5/ReadVariableOp(^highway_layer_2/MatMul_6/ReadVariableOp(^highway_layer_2/MatMul_7/ReadVariableOp%^highway_layer_2/add_1/ReadVariableOp%^highway_layer_2/add_3/ReadVariableOp%^highway_layer_2/add_5/ReadVariableOp%^highway_layer_2/add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2J
#highway_layer/MatMul/ReadVariableOp#highway_layer/MatMul/ReadVariableOp2N
%highway_layer/MatMul_1/ReadVariableOp%highway_layer/MatMul_1/ReadVariableOp2N
%highway_layer/MatMul_2/ReadVariableOp%highway_layer/MatMul_2/ReadVariableOp2N
%highway_layer/MatMul_3/ReadVariableOp%highway_layer/MatMul_3/ReadVariableOp2N
%highway_layer/MatMul_4/ReadVariableOp%highway_layer/MatMul_4/ReadVariableOp2N
%highway_layer/MatMul_5/ReadVariableOp%highway_layer/MatMul_5/ReadVariableOp2N
%highway_layer/MatMul_6/ReadVariableOp%highway_layer/MatMul_6/ReadVariableOp2N
%highway_layer/MatMul_7/ReadVariableOp%highway_layer/MatMul_7/ReadVariableOp2H
"highway_layer/add_1/ReadVariableOp"highway_layer/add_1/ReadVariableOp2H
"highway_layer/add_3/ReadVariableOp"highway_layer/add_3/ReadVariableOp2H
"highway_layer/add_5/ReadVariableOp"highway_layer/add_5/ReadVariableOp2H
"highway_layer/add_7/ReadVariableOp"highway_layer/add_7/ReadVariableOp2N
%highway_layer_1/MatMul/ReadVariableOp%highway_layer_1/MatMul/ReadVariableOp2R
'highway_layer_1/MatMul_1/ReadVariableOp'highway_layer_1/MatMul_1/ReadVariableOp2R
'highway_layer_1/MatMul_2/ReadVariableOp'highway_layer_1/MatMul_2/ReadVariableOp2R
'highway_layer_1/MatMul_3/ReadVariableOp'highway_layer_1/MatMul_3/ReadVariableOp2R
'highway_layer_1/MatMul_4/ReadVariableOp'highway_layer_1/MatMul_4/ReadVariableOp2R
'highway_layer_1/MatMul_5/ReadVariableOp'highway_layer_1/MatMul_5/ReadVariableOp2R
'highway_layer_1/MatMul_6/ReadVariableOp'highway_layer_1/MatMul_6/ReadVariableOp2R
'highway_layer_1/MatMul_7/ReadVariableOp'highway_layer_1/MatMul_7/ReadVariableOp2L
$highway_layer_1/add_1/ReadVariableOp$highway_layer_1/add_1/ReadVariableOp2L
$highway_layer_1/add_3/ReadVariableOp$highway_layer_1/add_3/ReadVariableOp2L
$highway_layer_1/add_5/ReadVariableOp$highway_layer_1/add_5/ReadVariableOp2L
$highway_layer_1/add_7/ReadVariableOp$highway_layer_1/add_7/ReadVariableOp2N
%highway_layer_2/MatMul/ReadVariableOp%highway_layer_2/MatMul/ReadVariableOp2R
'highway_layer_2/MatMul_1/ReadVariableOp'highway_layer_2/MatMul_1/ReadVariableOp2R
'highway_layer_2/MatMul_2/ReadVariableOp'highway_layer_2/MatMul_2/ReadVariableOp2R
'highway_layer_2/MatMul_3/ReadVariableOp'highway_layer_2/MatMul_3/ReadVariableOp2R
'highway_layer_2/MatMul_4/ReadVariableOp'highway_layer_2/MatMul_4/ReadVariableOp2R
'highway_layer_2/MatMul_5/ReadVariableOp'highway_layer_2/MatMul_5/ReadVariableOp2R
'highway_layer_2/MatMul_6/ReadVariableOp'highway_layer_2/MatMul_6/ReadVariableOp2R
'highway_layer_2/MatMul_7/ReadVariableOp'highway_layer_2/MatMul_7/ReadVariableOp2L
$highway_layer_2/add_1/ReadVariableOp$highway_layer_2/add_1/ReadVariableOp2L
$highway_layer_2/add_3/ReadVariableOp$highway_layer_2/add_3/ReadVariableOp2L
$highway_layer_2/add_5/ReadVariableOp$highway_layer_2/add_5/ReadVariableOp2L
$highway_layer_2/add_7/ReadVariableOp$highway_layer_2/add_7/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
+__inference_dpde_model_layer_call_fn_103153

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
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

unknown_20

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dpde_model_layer_call_and_return_conditional_losses_1020602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_highway_layer_2_layer_call_fn_103535$
 input_combined_original_variable!
input_combined_previous_layer
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall input_combined_original_variableinput_combined_previous_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_1016282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
'
_output_shapes
:?????????
:
_user_specified_name" input_combined/original_variable:fb
'
_output_shapes
:?????????Z
7
_user_specified_nameinput_combined/previous_layer
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_101695

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Z::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
+__inference_dpde_model_layer_call_fn_103254

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
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

unknown_20

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

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dpde_model_layer_call_and_return_conditional_losses_1023322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_101323

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?7
?
I__inference_highway_layer_layer_call_and_return_conditional_losses_101397
input_combined
input_combined_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource!
add_5_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource!
add_7_readvariableop_resource
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?add_1/ReadVariableOp?add_3/ReadVariableOp?add_5/ReadVariableOp?add_7/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOp{
MatMulMatMulinput_combinedMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMulinput_combined_1MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_1Q
TanhTanh	add_1:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMulinput_combinedMatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_2?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMulinput_combined_1MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_3q
add_2AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
add_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_3/ReadVariableOpr
add_3AddV2	add_2:z:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_1?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_4/ReadVariableOp?
MatMul_4MatMulinput_combinedMatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_5/ReadVariableOp?
MatMul_5MatMulinput_combined_1MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_5q
add_4AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
add_4?
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_2a
MulMulinput_combined_1
Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
Mul?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_6/ReadVariableOp?
MatMul_6MatMulinput_combinedMatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_7/ReadVariableOpz
MatMul_7MatMulMul:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_7q
add_6AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
add_6?
add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_7/ReadVariableOpr
add_7AddV2	add_6:z:0add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_7U
Tanh_3Tanh	add_7:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_3\
ones_like/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
	ones_likec
subSubones_like:output:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
sub\
Mul_1Mulsub:z:0
Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
Mul_1c
Mul_2MulTanh:y:0input_combined_1*
T0*'
_output_shapes
:?????????Z2
Mul_2_
add_8AddV2	Mul_1:z:0	Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
add_8?
IdentityIdentity	add_8:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^add_1/ReadVariableOp^add_3/ReadVariableOp^add_5/ReadVariableOp^add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp:W S
'
_output_shapes
:?????????
(
_user_specified_nameinput_combined:WS
'
_output_shapes
:?????????Z
(
_user_specified_nameinput_combined
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_103265

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_dpde_model_layer_call_and_return_conditional_losses_102797

inputs"
tf___operators___add_6_addv2_x"
tf___operators___add_2_addv2_x"
tf___operators___add_1_addv2_x 
tf___operators___add_addv2_x"
tf___operators___add_3_addv2_x
tf_math_multiply_6_mul_x
tf_math_multiply_7_mul_x(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource0
,highway_layer_matmul_readvariableop_resource2
.highway_layer_matmul_1_readvariableop_resource/
+highway_layer_add_1_readvariableop_resource2
.highway_layer_matmul_2_readvariableop_resource2
.highway_layer_matmul_3_readvariableop_resource/
+highway_layer_add_3_readvariableop_resource2
.highway_layer_matmul_4_readvariableop_resource2
.highway_layer_matmul_5_readvariableop_resource/
+highway_layer_add_5_readvariableop_resource2
.highway_layer_matmul_6_readvariableop_resource2
.highway_layer_matmul_7_readvariableop_resource/
+highway_layer_add_7_readvariableop_resource"
tf___operators___add_7_addv2_x2
.highway_layer_1_matmul_readvariableop_resource4
0highway_layer_1_matmul_1_readvariableop_resource1
-highway_layer_1_add_1_readvariableop_resource4
0highway_layer_1_matmul_2_readvariableop_resource4
0highway_layer_1_matmul_3_readvariableop_resource1
-highway_layer_1_add_3_readvariableop_resource4
0highway_layer_1_matmul_4_readvariableop_resource4
0highway_layer_1_matmul_5_readvariableop_resource1
-highway_layer_1_add_5_readvariableop_resource4
0highway_layer_1_matmul_6_readvariableop_resource4
0highway_layer_1_matmul_7_readvariableop_resource1
-highway_layer_1_add_7_readvariableop_resource2
.highway_layer_2_matmul_readvariableop_resource4
0highway_layer_2_matmul_1_readvariableop_resource1
-highway_layer_2_add_1_readvariableop_resource4
0highway_layer_2_matmul_2_readvariableop_resource4
0highway_layer_2_matmul_3_readvariableop_resource1
-highway_layer_2_add_3_readvariableop_resource4
0highway_layer_2_matmul_4_readvariableop_resource4
0highway_layer_2_matmul_5_readvariableop_resource1
-highway_layer_2_add_5_readvariableop_resource4
0highway_layer_2_matmul_6_readvariableop_resource4
0highway_layer_2_matmul_7_readvariableop_resource1
-highway_layer_2_add_7_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?#highway_layer/MatMul/ReadVariableOp?%highway_layer/MatMul_1/ReadVariableOp?%highway_layer/MatMul_2/ReadVariableOp?%highway_layer/MatMul_3/ReadVariableOp?%highway_layer/MatMul_4/ReadVariableOp?%highway_layer/MatMul_5/ReadVariableOp?%highway_layer/MatMul_6/ReadVariableOp?%highway_layer/MatMul_7/ReadVariableOp?"highway_layer/add_1/ReadVariableOp?"highway_layer/add_3/ReadVariableOp?"highway_layer/add_5/ReadVariableOp?"highway_layer/add_7/ReadVariableOp?%highway_layer_1/MatMul/ReadVariableOp?'highway_layer_1/MatMul_1/ReadVariableOp?'highway_layer_1/MatMul_2/ReadVariableOp?'highway_layer_1/MatMul_3/ReadVariableOp?'highway_layer_1/MatMul_4/ReadVariableOp?'highway_layer_1/MatMul_5/ReadVariableOp?'highway_layer_1/MatMul_6/ReadVariableOp?'highway_layer_1/MatMul_7/ReadVariableOp?$highway_layer_1/add_1/ReadVariableOp?$highway_layer_1/add_3/ReadVariableOp?$highway_layer_1/add_5/ReadVariableOp?$highway_layer_1/add_7/ReadVariableOp?%highway_layer_2/MatMul/ReadVariableOp?'highway_layer_2/MatMul_1/ReadVariableOp?'highway_layer_2/MatMul_2/ReadVariableOp?'highway_layer_2/MatMul_3/ReadVariableOp?'highway_layer_2/MatMul_4/ReadVariableOp?'highway_layer_2/MatMul_5/ReadVariableOp?'highway_layer_2/MatMul_6/ReadVariableOp?'highway_layer_2/MatMul_7/ReadVariableOp?$highway_layer_2/add_1/ReadVariableOp?$highway_layer_2/add_3/ReadVariableOp?$highway_layer_2/add_5/ReadVariableOp?$highway_layer_2/add_7/ReadVariableOp?
.tf.__operators__.getitem_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_4/strided_slice/stack?
0tf.__operators__.getitem_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_4/strided_slice/stack_1?
0tf.__operators__.getitem_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_4/strided_slice/stack_2?
(tf.__operators__.getitem_4/strided_sliceStridedSliceinputs7tf.__operators__.getitem_4/strided_slice/stack:output:09tf.__operators__.getitem_4/strided_slice/stack_1:output:09tf.__operators__.getitem_4/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_4/strided_slice?
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_2/strided_slice/stack?
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_2/strided_slice/stack_1?
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_2/strided_slice/stack_2?
(tf.__operators__.getitem_2/strided_sliceStridedSliceinputs7tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_2/strided_slice?
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_1/strided_slice/stack?
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_1/strided_slice/stack_1?
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_1/strided_slice/stack_2?
(tf.__operators__.getitem_1/strided_sliceStridedSliceinputs7tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_1/strided_slice?
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,tf.__operators__.getitem/strided_slice/stack?
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem/strided_slice/stack_1?
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.tf.__operators__.getitem/strided_slice/stack_2?
&tf.__operators__.getitem/strided_sliceStridedSliceinputs5tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2(
&tf.__operators__.getitem/strided_slicey
tf.math.subtract_4/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_4/Sub/y?
tf.math.subtract_4/SubSub1tf.__operators__.getitem_4/strided_slice:output:0!tf.math.subtract_4/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_4/Sub?
.tf.__operators__.getitem_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       20
.tf.__operators__.getitem_3/strided_slice/stack?
0tf.__operators__.getitem_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0tf.__operators__.getitem_3/strided_slice/stack_1?
0tf.__operators__.getitem_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0tf.__operators__.getitem_3/strided_slice/stack_2?
(tf.__operators__.getitem_3/strided_sliceStridedSliceinputs7tf.__operators__.getitem_3/strided_slice/stack:output:09tf.__operators__.getitem_3/strided_slice/stack_1:output:09tf.__operators__.getitem_3/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2*
(tf.__operators__.getitem_3/strided_slicey
tf.math.subtract_2/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_2/Sub/y?
tf.math.subtract_2/SubSub1tf.__operators__.getitem_2/strided_slice:output:0!tf.math.subtract_2/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_2/Suby
tf.math.subtract_1/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_1/Sub/y?
tf.math.subtract_1/SubSub1tf.__operators__.getitem_1/strided_slice:output:0!tf.math.subtract_1/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_1/Subu
tf.math.subtract/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract/Sub/y?
tf.math.subtract/SubSub/tf.__operators__.getitem/strided_slice:output:0tf.math.subtract/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract/Suby
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMultf.math.subtract_4/Sub:z:0!tf.math.multiply_4/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_4/Muly
tf.math.subtract_3/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
tf.math.subtract_3/Sub/y?
tf.math.subtract_3/SubSub1tf.__operators__.getitem_3/strided_slice:output:0!tf.math.subtract_3/Sub/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_3/Suby
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMultf.math.subtract_2/Sub:z:0!tf.math.multiply_2/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_2/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMultf.math.subtract_1/Sub:z:0!tf.math.multiply_1/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_1/Mulu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?@2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMultf.math.subtract/Sub:z:0tf.math.multiply/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply/Mul
tf.math.truediv_5/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_5/truediv/y?
tf.math.truediv_5/truedivRealDivtf.math.multiply_4/Mul:z:0$tf.math.truediv_5/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_5/truedivy
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *?UC@2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMultf.math.subtract_3/Sub:z:0!tf.math.multiply_3/Mul/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_3/Mul
tf.math.truediv_2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_2/truediv/y?
tf.math.truediv_2/truedivRealDivtf.math.multiply_2/Mul:z:0$tf.math.truediv_2/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_2/truediv
tf.math.truediv_1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_1/truediv/y?
tf.math.truediv_1/truedivRealDivtf.math.multiply_1/Mul:z:0$tf.math.truediv_1/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_1/truediv{
tf.math.truediv/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv/truediv/y?
tf.math.truediv/truedivRealDivtf.math.multiply/Mul:z:0"tf.math.truediv/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv/truediv?
tf.__operators__.add_6/AddV2AddV2tf___operators___add_6_addv2_xtf.math.truediv_5/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_6/AddV2
tf.math.truediv_3/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
tf.math.truediv_3/truediv/y?
tf.math.truediv_3/truedivRealDivtf.math.multiply_3/Mul:z:0$tf.math.truediv_3/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_3/truediv?
tf.__operators__.add_2/AddV2AddV2tf___operators___add_2_addv2_xtf.math.truediv_2/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_2/AddV2?
tf.__operators__.add_1/AddV2AddV2tf___operators___add_1_addv2_xtf.math.truediv_1/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_1/AddV2?
tf.math.negative/NegNeg tf.__operators__.add_6/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.negative/Neg?
tf.__operators__.add/AddV2AddV2tf___operators___add_addv2_xtf.math.truediv/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add/AddV2?
tf.__operators__.add_3/AddV2AddV2tf___operators___add_3_addv2_xtf.math.truediv_3/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_3/AddV2}
tf.math.exp/ExpExp tf.__operators__.add_1/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp/Exp?
tf.math.exp_1/ExpExp tf.__operators__.add_2/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_1/Exp?
tf.math.multiply_5/MulMultf.math.negative/Neg:y:0tf.__operators__.add/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_5/Mul?
tf.__operators__.add_4/AddV2AddV2tf.math.exp/Exp:y:0tf.math.exp_1/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_4/AddV2?
tf.math.exp_2/ExpExp tf.__operators__.add_3/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_2/Exp{
tf.math.exp_3/ExpExptf.math.multiply_5/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_3/Exp?
tf.__operators__.add_5/AddV2AddV2 tf.__operators__.add_4/AddV2:z:0tf.math.exp_2/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_5/AddV2
tf.math.truediv_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
tf.math.truediv_4/truediv/y?
tf.math.truediv_4/truedivRealDiv tf.__operators__.add_5/AddV2:z:0$tf.math.truediv_4/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_4/truediv?
tf.math.multiply_6/MulMultf_math_multiply_6_mul_xtf.math.exp_3/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_6/Mul?
tf.math.subtract_5/SubSubtf.math.truediv_4/truediv:z:0tf.math.multiply_6/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.subtract_5/Sub?
tf.math.multiply_7/MulMultf_math_multiply_7_mul_xtf.math.subtract_5/Sub:z:0*
T0*'
_output_shapes
:?????????2
tf.math.multiply_7/Mul?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
dense/BiasAddj

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2

dense/Tanh{
tf.math.exp_4/ExpExptf.math.multiply_7/Mul:z:0*
T0*'
_output_shapes
:?????????2
tf.math.exp_4/Exp?
#highway_layer/MatMul/ReadVariableOpReadVariableOp,highway_layer_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02%
#highway_layer/MatMul/ReadVariableOp?
highway_layer/MatMulMatMulinputs+highway_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul?
%highway_layer/MatMul_1/ReadVariableOpReadVariableOp.highway_layer_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_1/ReadVariableOp?
highway_layer/MatMul_1MatMuldense/Tanh:y:0-highway_layer/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_1?
highway_layer/addAddV2highway_layer/MatMul:product:0 highway_layer/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add?
"highway_layer/add_1/ReadVariableOpReadVariableOp+highway_layer_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_1/ReadVariableOp?
highway_layer/add_1AddV2highway_layer/add:z:0*highway_layer/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_1{
highway_layer/TanhTanhhighway_layer/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh?
%highway_layer/MatMul_2/ReadVariableOpReadVariableOp.highway_layer_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer/MatMul_2/ReadVariableOp?
highway_layer/MatMul_2MatMulinputs-highway_layer/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_2?
%highway_layer/MatMul_3/ReadVariableOpReadVariableOp.highway_layer_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_3/ReadVariableOp?
highway_layer/MatMul_3MatMuldense/Tanh:y:0-highway_layer/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_3?
highway_layer/add_2AddV2 highway_layer/MatMul_2:product:0 highway_layer/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_2?
"highway_layer/add_3/ReadVariableOpReadVariableOp+highway_layer_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_3/ReadVariableOp?
highway_layer/add_3AddV2highway_layer/add_2:z:0*highway_layer/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_3
highway_layer/Tanh_1Tanhhighway_layer/add_3:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh_1?
%highway_layer/MatMul_4/ReadVariableOpReadVariableOp.highway_layer_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer/MatMul_4/ReadVariableOp?
highway_layer/MatMul_4MatMulinputs-highway_layer/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_4?
%highway_layer/MatMul_5/ReadVariableOpReadVariableOp.highway_layer_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_5/ReadVariableOp?
highway_layer/MatMul_5MatMuldense/Tanh:y:0-highway_layer/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_5?
highway_layer/add_4AddV2 highway_layer/MatMul_4:product:0 highway_layer/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_4?
"highway_layer/add_5/ReadVariableOpReadVariableOp+highway_layer_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_5/ReadVariableOp?
highway_layer/add_5AddV2highway_layer/add_4:z:0*highway_layer/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_5
highway_layer/Tanh_2Tanhhighway_layer/add_5:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh_2?
highway_layer/MulMuldense/Tanh:y:0highway_layer/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Mul?
%highway_layer/MatMul_6/ReadVariableOpReadVariableOp.highway_layer_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer/MatMul_6/ReadVariableOp?
highway_layer/MatMul_6MatMulinputs-highway_layer/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_6?
%highway_layer/MatMul_7/ReadVariableOpReadVariableOp.highway_layer_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02'
%highway_layer/MatMul_7/ReadVariableOp?
highway_layer/MatMul_7MatMulhighway_layer/Mul:z:0-highway_layer/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/MatMul_7?
highway_layer/add_6AddV2 highway_layer/MatMul_6:product:0 highway_layer/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_6?
"highway_layer/add_7/ReadVariableOpReadVariableOp+highway_layer_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02$
"highway_layer/add_7/ReadVariableOp?
highway_layer/add_7AddV2highway_layer/add_6:z:0*highway_layer/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_7
highway_layer/Tanh_3Tanhhighway_layer/add_7:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Tanh_3?
highway_layer/ones_like/ShapeShapehighway_layer/Tanh_1:y:0*
T0*
_output_shapes
:2
highway_layer/ones_like/Shape?
highway_layer/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
highway_layer/ones_like/Const?
highway_layer/ones_likeFill&highway_layer/ones_like/Shape:output:0&highway_layer/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/ones_like?
highway_layer/subSub highway_layer/ones_like:output:0highway_layer/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/sub?
highway_layer/Mul_1Mulhighway_layer/sub:z:0highway_layer/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Mul_1?
highway_layer/Mul_2Mulhighway_layer/Tanh:y:0dense/Tanh:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/Mul_2?
highway_layer/add_8AddV2highway_layer/Mul_1:z:0highway_layer/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer/add_8?
tf.__operators__.add_7/AddV2AddV2tf___operators___add_7_addv2_xtf.math.exp_4/Exp:y:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_7/AddV2?
%highway_layer_1/MatMul/ReadVariableOpReadVariableOp.highway_layer_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer_1/MatMul/ReadVariableOp?
highway_layer_1/MatMulMatMulinputs-highway_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul?
'highway_layer_1/MatMul_1/ReadVariableOpReadVariableOp0highway_layer_1_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_1/ReadVariableOp?
highway_layer_1/MatMul_1MatMulhighway_layer/add_8:z:0/highway_layer_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_1?
highway_layer_1/addAddV2 highway_layer_1/MatMul:product:0"highway_layer_1/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add?
$highway_layer_1/add_1/ReadVariableOpReadVariableOp-highway_layer_1_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_1/ReadVariableOp?
highway_layer_1/add_1AddV2highway_layer_1/add:z:0,highway_layer_1/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_1?
highway_layer_1/TanhTanhhighway_layer_1/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh?
'highway_layer_1/MatMul_2/ReadVariableOpReadVariableOp0highway_layer_1_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_1/MatMul_2/ReadVariableOp?
highway_layer_1/MatMul_2MatMulinputs/highway_layer_1/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_2?
'highway_layer_1/MatMul_3/ReadVariableOpReadVariableOp0highway_layer_1_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_3/ReadVariableOp?
highway_layer_1/MatMul_3MatMulhighway_layer/add_8:z:0/highway_layer_1/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_3?
highway_layer_1/add_2AddV2"highway_layer_1/MatMul_2:product:0"highway_layer_1/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_2?
$highway_layer_1/add_3/ReadVariableOpReadVariableOp-highway_layer_1_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_3/ReadVariableOp?
highway_layer_1/add_3AddV2highway_layer_1/add_2:z:0,highway_layer_1/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_3?
highway_layer_1/Tanh_1Tanhhighway_layer_1/add_3:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh_1?
'highway_layer_1/MatMul_4/ReadVariableOpReadVariableOp0highway_layer_1_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_1/MatMul_4/ReadVariableOp?
highway_layer_1/MatMul_4MatMulinputs/highway_layer_1/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_4?
'highway_layer_1/MatMul_5/ReadVariableOpReadVariableOp0highway_layer_1_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_5/ReadVariableOp?
highway_layer_1/MatMul_5MatMulhighway_layer/add_8:z:0/highway_layer_1/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_5?
highway_layer_1/add_4AddV2"highway_layer_1/MatMul_4:product:0"highway_layer_1/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_4?
$highway_layer_1/add_5/ReadVariableOpReadVariableOp-highway_layer_1_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_5/ReadVariableOp?
highway_layer_1/add_5AddV2highway_layer_1/add_4:z:0,highway_layer_1/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_5?
highway_layer_1/Tanh_2Tanhhighway_layer_1/add_5:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh_2?
highway_layer_1/MulMulhighway_layer/add_8:z:0highway_layer_1/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Mul?
'highway_layer_1/MatMul_6/ReadVariableOpReadVariableOp0highway_layer_1_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_1/MatMul_6/ReadVariableOp?
highway_layer_1/MatMul_6MatMulinputs/highway_layer_1/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_6?
'highway_layer_1/MatMul_7/ReadVariableOpReadVariableOp0highway_layer_1_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_1/MatMul_7/ReadVariableOp?
highway_layer_1/MatMul_7MatMulhighway_layer_1/Mul:z:0/highway_layer_1/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/MatMul_7?
highway_layer_1/add_6AddV2"highway_layer_1/MatMul_6:product:0"highway_layer_1/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_6?
$highway_layer_1/add_7/ReadVariableOpReadVariableOp-highway_layer_1_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_1/add_7/ReadVariableOp?
highway_layer_1/add_7AddV2highway_layer_1/add_6:z:0,highway_layer_1/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_7?
highway_layer_1/Tanh_3Tanhhighway_layer_1/add_7:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Tanh_3?
highway_layer_1/ones_like/ShapeShapehighway_layer_1/Tanh_1:y:0*
T0*
_output_shapes
:2!
highway_layer_1/ones_like/Shape?
highway_layer_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
highway_layer_1/ones_like/Const?
highway_layer_1/ones_likeFill(highway_layer_1/ones_like/Shape:output:0(highway_layer_1/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/ones_like?
highway_layer_1/subSub"highway_layer_1/ones_like:output:0highway_layer_1/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/sub?
highway_layer_1/Mul_1Mulhighway_layer_1/sub:z:0highway_layer_1/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Mul_1?
highway_layer_1/Mul_2Mulhighway_layer_1/Tanh:y:0highway_layer/add_8:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/Mul_2?
highway_layer_1/add_8AddV2highway_layer_1/Mul_1:z:0highway_layer_1/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_1/add_8}
tf.math.log/LogLog tf.__operators__.add_7/AddV2:z:0*
T0*'
_output_shapes
:?????????2
tf.math.log/Log?
%highway_layer_2/MatMul/ReadVariableOpReadVariableOp.highway_layer_2_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02'
%highway_layer_2/MatMul/ReadVariableOp?
highway_layer_2/MatMulMatMulinputs-highway_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul?
'highway_layer_2/MatMul_1/ReadVariableOpReadVariableOp0highway_layer_2_matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_1/ReadVariableOp?
highway_layer_2/MatMul_1MatMulhighway_layer_1/add_8:z:0/highway_layer_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_1?
highway_layer_2/addAddV2 highway_layer_2/MatMul:product:0"highway_layer_2/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add?
$highway_layer_2/add_1/ReadVariableOpReadVariableOp-highway_layer_2_add_1_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_1/ReadVariableOp?
highway_layer_2/add_1AddV2highway_layer_2/add:z:0,highway_layer_2/add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_1?
highway_layer_2/TanhTanhhighway_layer_2/add_1:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh?
'highway_layer_2/MatMul_2/ReadVariableOpReadVariableOp0highway_layer_2_matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_2/MatMul_2/ReadVariableOp?
highway_layer_2/MatMul_2MatMulinputs/highway_layer_2/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_2?
'highway_layer_2/MatMul_3/ReadVariableOpReadVariableOp0highway_layer_2_matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_3/ReadVariableOp?
highway_layer_2/MatMul_3MatMulhighway_layer_1/add_8:z:0/highway_layer_2/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_3?
highway_layer_2/add_2AddV2"highway_layer_2/MatMul_2:product:0"highway_layer_2/MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_2?
$highway_layer_2/add_3/ReadVariableOpReadVariableOp-highway_layer_2_add_3_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_3/ReadVariableOp?
highway_layer_2/add_3AddV2highway_layer_2/add_2:z:0,highway_layer_2/add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_3?
highway_layer_2/Tanh_1Tanhhighway_layer_2/add_3:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh_1?
'highway_layer_2/MatMul_4/ReadVariableOpReadVariableOp0highway_layer_2_matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_2/MatMul_4/ReadVariableOp?
highway_layer_2/MatMul_4MatMulinputs/highway_layer_2/MatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_4?
'highway_layer_2/MatMul_5/ReadVariableOpReadVariableOp0highway_layer_2_matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_5/ReadVariableOp?
highway_layer_2/MatMul_5MatMulhighway_layer_1/add_8:z:0/highway_layer_2/MatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_5?
highway_layer_2/add_4AddV2"highway_layer_2/MatMul_4:product:0"highway_layer_2/MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_4?
$highway_layer_2/add_5/ReadVariableOpReadVariableOp-highway_layer_2_add_5_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_5/ReadVariableOp?
highway_layer_2/add_5AddV2highway_layer_2/add_4:z:0,highway_layer_2/add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_5?
highway_layer_2/Tanh_2Tanhhighway_layer_2/add_5:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh_2?
highway_layer_2/MulMulhighway_layer_1/add_8:z:0highway_layer_2/Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Mul?
'highway_layer_2/MatMul_6/ReadVariableOpReadVariableOp0highway_layer_2_matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02)
'highway_layer_2/MatMul_6/ReadVariableOp?
highway_layer_2/MatMul_6MatMulinputs/highway_layer_2/MatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_6?
'highway_layer_2/MatMul_7/ReadVariableOpReadVariableOp0highway_layer_2_matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02)
'highway_layer_2/MatMul_7/ReadVariableOp?
highway_layer_2/MatMul_7MatMulhighway_layer_2/Mul:z:0/highway_layer_2/MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/MatMul_7?
highway_layer_2/add_6AddV2"highway_layer_2/MatMul_6:product:0"highway_layer_2/MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_6?
$highway_layer_2/add_7/ReadVariableOpReadVariableOp-highway_layer_2_add_7_readvariableop_resource*
_output_shapes
:Z*
dtype02&
$highway_layer_2/add_7/ReadVariableOp?
highway_layer_2/add_7AddV2highway_layer_2/add_6:z:0,highway_layer_2/add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_7?
highway_layer_2/Tanh_3Tanhhighway_layer_2/add_7:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Tanh_3?
highway_layer_2/ones_like/ShapeShapehighway_layer_2/Tanh_1:y:0*
T0*
_output_shapes
:2!
highway_layer_2/ones_like/Shape?
highway_layer_2/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
highway_layer_2/ones_like/Const?
highway_layer_2/ones_likeFill(highway_layer_2/ones_like/Shape:output:0(highway_layer_2/ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/ones_like?
highway_layer_2/subSub"highway_layer_2/ones_like:output:0highway_layer_2/Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/sub?
highway_layer_2/Mul_1Mulhighway_layer_2/sub:z:0highway_layer_2/Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Mul_1?
highway_layer_2/Mul_2Mulhighway_layer_2/Tanh:y:0highway_layer_1/add_8:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/Mul_2?
highway_layer_2/add_8AddV2highway_layer_2/Mul_1:z:0highway_layer_2/Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
highway_layer_2/add_8?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulhighway_layer_2/add_8:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd
tf.math.truediv_6/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
tf.math.truediv_6/truediv/y?
tf.math.truediv_6/truedivRealDivtf.math.log/Log:y:0$tf.math.truediv_6/truediv/y:output:0*
T0*'
_output_shapes
:?????????2
tf.math.truediv_6/truediv?
tf.__operators__.add_8/AddV2AddV2dense_1/BiasAdd:output:0tf.math.truediv_6/truediv:z:0*
T0*'
_output_shapes
:?????????2
tf.__operators__.add_8/AddV2?
IdentityIdentity tf.__operators__.add_8/AddV2:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp$^highway_layer/MatMul/ReadVariableOp&^highway_layer/MatMul_1/ReadVariableOp&^highway_layer/MatMul_2/ReadVariableOp&^highway_layer/MatMul_3/ReadVariableOp&^highway_layer/MatMul_4/ReadVariableOp&^highway_layer/MatMul_5/ReadVariableOp&^highway_layer/MatMul_6/ReadVariableOp&^highway_layer/MatMul_7/ReadVariableOp#^highway_layer/add_1/ReadVariableOp#^highway_layer/add_3/ReadVariableOp#^highway_layer/add_5/ReadVariableOp#^highway_layer/add_7/ReadVariableOp&^highway_layer_1/MatMul/ReadVariableOp(^highway_layer_1/MatMul_1/ReadVariableOp(^highway_layer_1/MatMul_2/ReadVariableOp(^highway_layer_1/MatMul_3/ReadVariableOp(^highway_layer_1/MatMul_4/ReadVariableOp(^highway_layer_1/MatMul_5/ReadVariableOp(^highway_layer_1/MatMul_6/ReadVariableOp(^highway_layer_1/MatMul_7/ReadVariableOp%^highway_layer_1/add_1/ReadVariableOp%^highway_layer_1/add_3/ReadVariableOp%^highway_layer_1/add_5/ReadVariableOp%^highway_layer_1/add_7/ReadVariableOp&^highway_layer_2/MatMul/ReadVariableOp(^highway_layer_2/MatMul_1/ReadVariableOp(^highway_layer_2/MatMul_2/ReadVariableOp(^highway_layer_2/MatMul_3/ReadVariableOp(^highway_layer_2/MatMul_4/ReadVariableOp(^highway_layer_2/MatMul_5/ReadVariableOp(^highway_layer_2/MatMul_6/ReadVariableOp(^highway_layer_2/MatMul_7/ReadVariableOp%^highway_layer_2/add_1/ReadVariableOp%^highway_layer_2/add_3/ReadVariableOp%^highway_layer_2/add_5/ReadVariableOp%^highway_layer_2/add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????: : : : : : : ::::::::::::::: ::::::::::::::::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2J
#highway_layer/MatMul/ReadVariableOp#highway_layer/MatMul/ReadVariableOp2N
%highway_layer/MatMul_1/ReadVariableOp%highway_layer/MatMul_1/ReadVariableOp2N
%highway_layer/MatMul_2/ReadVariableOp%highway_layer/MatMul_2/ReadVariableOp2N
%highway_layer/MatMul_3/ReadVariableOp%highway_layer/MatMul_3/ReadVariableOp2N
%highway_layer/MatMul_4/ReadVariableOp%highway_layer/MatMul_4/ReadVariableOp2N
%highway_layer/MatMul_5/ReadVariableOp%highway_layer/MatMul_5/ReadVariableOp2N
%highway_layer/MatMul_6/ReadVariableOp%highway_layer/MatMul_6/ReadVariableOp2N
%highway_layer/MatMul_7/ReadVariableOp%highway_layer/MatMul_7/ReadVariableOp2H
"highway_layer/add_1/ReadVariableOp"highway_layer/add_1/ReadVariableOp2H
"highway_layer/add_3/ReadVariableOp"highway_layer/add_3/ReadVariableOp2H
"highway_layer/add_5/ReadVariableOp"highway_layer/add_5/ReadVariableOp2H
"highway_layer/add_7/ReadVariableOp"highway_layer/add_7/ReadVariableOp2N
%highway_layer_1/MatMul/ReadVariableOp%highway_layer_1/MatMul/ReadVariableOp2R
'highway_layer_1/MatMul_1/ReadVariableOp'highway_layer_1/MatMul_1/ReadVariableOp2R
'highway_layer_1/MatMul_2/ReadVariableOp'highway_layer_1/MatMul_2/ReadVariableOp2R
'highway_layer_1/MatMul_3/ReadVariableOp'highway_layer_1/MatMul_3/ReadVariableOp2R
'highway_layer_1/MatMul_4/ReadVariableOp'highway_layer_1/MatMul_4/ReadVariableOp2R
'highway_layer_1/MatMul_5/ReadVariableOp'highway_layer_1/MatMul_5/ReadVariableOp2R
'highway_layer_1/MatMul_6/ReadVariableOp'highway_layer_1/MatMul_6/ReadVariableOp2R
'highway_layer_1/MatMul_7/ReadVariableOp'highway_layer_1/MatMul_7/ReadVariableOp2L
$highway_layer_1/add_1/ReadVariableOp$highway_layer_1/add_1/ReadVariableOp2L
$highway_layer_1/add_3/ReadVariableOp$highway_layer_1/add_3/ReadVariableOp2L
$highway_layer_1/add_5/ReadVariableOp$highway_layer_1/add_5/ReadVariableOp2L
$highway_layer_1/add_7/ReadVariableOp$highway_layer_1/add_7/ReadVariableOp2N
%highway_layer_2/MatMul/ReadVariableOp%highway_layer_2/MatMul/ReadVariableOp2R
'highway_layer_2/MatMul_1/ReadVariableOp'highway_layer_2/MatMul_1/ReadVariableOp2R
'highway_layer_2/MatMul_2/ReadVariableOp'highway_layer_2/MatMul_2/ReadVariableOp2R
'highway_layer_2/MatMul_3/ReadVariableOp'highway_layer_2/MatMul_3/ReadVariableOp2R
'highway_layer_2/MatMul_4/ReadVariableOp'highway_layer_2/MatMul_4/ReadVariableOp2R
'highway_layer_2/MatMul_5/ReadVariableOp'highway_layer_2/MatMul_5/ReadVariableOp2R
'highway_layer_2/MatMul_6/ReadVariableOp'highway_layer_2/MatMul_6/ReadVariableOp2R
'highway_layer_2/MatMul_7/ReadVariableOp'highway_layer_2/MatMul_7/ReadVariableOp2L
$highway_layer_2/add_1/ReadVariableOp$highway_layer_2/add_1/ReadVariableOp2L
$highway_layer_2/add_3/ReadVariableOp$highway_layer_2/add_3/ReadVariableOp2L
$highway_layer_2/add_5/ReadVariableOp$highway_layer_2/add_5/ReadVariableOp2L
$highway_layer_2/add_7/ReadVariableOp$highway_layer_2/add_7/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?8
?
I__inference_highway_layer_layer_call_and_return_conditional_losses_103331$
 input_combined_original_variable!
input_combined_previous_layer"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource!
add_5_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource!
add_7_readvariableop_resource
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?add_1/ReadVariableOp?add_3/ReadVariableOp?add_5/ReadVariableOp?add_7/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul input_combined_original_variableMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMulinput_combined_previous_layerMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_1Q
TanhTanh	add_1:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMul input_combined_original_variableMatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_2?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMulinput_combined_previous_layerMatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_3q
add_2AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
add_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_3/ReadVariableOpr
add_3AddV2	add_2:z:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_1?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_4/ReadVariableOp?
MatMul_4MatMul input_combined_original_variableMatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_5/ReadVariableOp?
MatMul_5MatMulinput_combined_previous_layerMatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_5q
add_4AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
add_4?
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_2n
MulMulinput_combined_previous_layer
Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
Mul?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_6/ReadVariableOp?
MatMul_6MatMul input_combined_original_variableMatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_7/ReadVariableOpz
MatMul_7MatMulMul:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_7q
add_6AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
add_6?
add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_7/ReadVariableOpr
add_7AddV2	add_6:z:0add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_7U
Tanh_3Tanh	add_7:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_3\
ones_like/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
	ones_likec
subSubones_like:output:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
sub\
Mul_1Mulsub:z:0
Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
Mul_1p
Mul_2MulTanh:y:0input_combined_previous_layer*
T0*'
_output_shapes
:?????????Z2
Mul_2_
add_8AddV2	Mul_1:z:0	Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
add_8?
IdentityIdentity	add_8:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^add_1/ReadVariableOp^add_3/ReadVariableOp^add_5/ReadVariableOp^add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp:i e
'
_output_shapes
:?????????
:
_user_specified_name" input_combined/original_variable:fb
'
_output_shapes
:?????????Z
7
_user_specified_nameinput_combined/previous_layer
?
?
0__inference_highway_layer_1_layer_call_fn_103448$
 input_combined_original_variable!
input_combined_previous_layer
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

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall input_combined_original_variableinput_combined_previous_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_1015132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
'
_output_shapes
:?????????
:
_user_specified_name" input_combined/original_variable:fb
'
_output_shapes
:?????????Z
7
_user_specified_nameinput_combined/previous_layer
?8
?
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_103418$
 input_combined_original_variable!
input_combined_previous_layer"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource!
add_1_readvariableop_resource$
 matmul_2_readvariableop_resource$
 matmul_3_readvariableop_resource!
add_3_readvariableop_resource$
 matmul_4_readvariableop_resource$
 matmul_5_readvariableop_resource!
add_5_readvariableop_resource$
 matmul_6_readvariableop_resource$
 matmul_7_readvariableop_resource!
add_7_readvariableop_resource
identity??MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?MatMul_2/ReadVariableOp?MatMul_3/ReadVariableOp?MatMul_4/ReadVariableOp?MatMul_5/ReadVariableOp?MatMul_6/ReadVariableOp?MatMul_7/ReadVariableOp?add_1/ReadVariableOp?add_3/ReadVariableOp?add_5/ReadVariableOp?add_7/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOp?
MatMulMatMul input_combined_original_variableMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_1/ReadVariableOp?
MatMul_1MatMulinput_combined_previous_layerMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????Z2
add?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_1/ReadVariableOpp
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_1Q
TanhTanh	add_1:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_2/ReadVariableOp?
MatMul_2MatMul input_combined_original_variableMatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_2?
MatMul_3/ReadVariableOpReadVariableOp matmul_3_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_3/ReadVariableOp?
MatMul_3MatMulinput_combined_previous_layerMatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_3q
add_2AddV2MatMul_2:product:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????Z2
add_2?
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_3/ReadVariableOpr
add_3AddV2	add_2:z:0add_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_3U
Tanh_1Tanh	add_3:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_1?
MatMul_4/ReadVariableOpReadVariableOp matmul_4_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_4/ReadVariableOp?
MatMul_4MatMul input_combined_original_variableMatMul_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_4?
MatMul_5/ReadVariableOpReadVariableOp matmul_5_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_5/ReadVariableOp?
MatMul_5MatMulinput_combined_previous_layerMatMul_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_5q
add_4AddV2MatMul_4:product:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????Z2
add_4?
add_5/ReadVariableOpReadVariableOpadd_5_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_5/ReadVariableOpr
add_5AddV2	add_4:z:0add_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_5U
Tanh_2Tanh	add_5:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_2n
MulMulinput_combined_previous_layer
Tanh_2:y:0*
T0*'
_output_shapes
:?????????Z2
Mul?
MatMul_6/ReadVariableOpReadVariableOp matmul_6_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul_6/ReadVariableOp?
MatMul_6MatMul input_combined_original_variableMatMul_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_6?
MatMul_7/ReadVariableOpReadVariableOp matmul_7_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul_7/ReadVariableOpz
MatMul_7MatMulMul:z:0MatMul_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2

MatMul_7q
add_6AddV2MatMul_6:product:0MatMul_7:product:0*
T0*'
_output_shapes
:?????????Z2
add_6?
add_7/ReadVariableOpReadVariableOpadd_7_readvariableop_resource*
_output_shapes
:Z*
dtype02
add_7/ReadVariableOpr
add_7AddV2	add_6:z:0add_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
add_7U
Tanh_3Tanh	add_7:z:0*
T0*'
_output_shapes
:?????????Z2
Tanh_3\
ones_like/ShapeShape
Tanh_1:y:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:?????????Z2
	ones_likec
subSubones_like:output:0
Tanh_1:y:0*
T0*'
_output_shapes
:?????????Z2
sub\
Mul_1Mulsub:z:0
Tanh_3:y:0*
T0*'
_output_shapes
:?????????Z2
Mul_1p
Mul_2MulTanh:y:0input_combined_previous_layer*
T0*'
_output_shapes
:?????????Z2
Mul_2_
add_8AddV2	Mul_1:z:0	Mul_2:z:0*
T0*'
_output_shapes
:?????????Z2
add_8?
IdentityIdentity	add_8:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^MatMul_4/ReadVariableOp^MatMul_5/ReadVariableOp^MatMul_6/ReadVariableOp^MatMul_7/ReadVariableOp^add_1/ReadVariableOp^add_3/ReadVariableOp^add_5/ReadVariableOp^add_7/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????Z::::::::::::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp22
MatMul_4/ReadVariableOpMatMul_4/ReadVariableOp22
MatMul_5/ReadVariableOpMatMul_5/ReadVariableOp22
MatMul_6/ReadVariableOpMatMul_6/ReadVariableOp22
MatMul_7/ReadVariableOpMatMul_7/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp:i e
'
_output_shapes
:?????????
:
_user_specified_name" input_combined/original_variable:fb
'
_output_shapes
:?????????Z
7
_user_specified_nameinput_combined/previous_layer"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????J
tf.__operators__.add_80
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ȣ
?v
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer-33
#layer-34
$layer-35
%layer-36
&layer_with_weights-0
&layer-37
'layer-38
(layer_with_weights-1
(layer-39
)layer-40
*layer_with_weights-2
*layer-41
+layer-42
,layer_with_weights-3
,layer-43
-layer-44
.layer_with_weights-4
.layer-45
/layer-46
0layer-47
1	optimizer
2loss
3trainable_variables
4regularization_losses
5	variables
6	keras_api
7
signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?n
_tf_keras_network?n{"class_name": "DPDEModel", "name": "dpde_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "dpde_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem_1", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem_1", "inbound_nodes": [["input_1", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"start": null, "stop": null, "step": null}, {"start": 1, "stop": 2, "step": null}]}}]]}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem_2", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem_2", "inbound_nodes": [["input_1", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"start": null, "stop": null, "step": null}, {"start": 2, "stop": 3, "step": null}]}}]]}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem_4", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem_4", "inbound_nodes": [["input_1", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"start": null, "stop": null, "step": null}, {"start": 4, "stop": 5, "step": null}]}}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["tf.__operators__.getitem_1", 0, 0, {"y": -1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_2", "inbound_nodes": [["tf.__operators__.getitem_2", 0, 0, {"y": -1, "name": null}]]}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem_3", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem_3", "inbound_nodes": [["input_1", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"start": null, "stop": null, "step": null}, {"start": 3, "stop": 4, "step": null}]}}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_4", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_4", "inbound_nodes": [["tf.__operators__.getitem_4", 0, 0, {"y": -1, "name": null}]]}, {"class_name": "SlicingOpLambda", "config": {"name": "tf.__operators__.getitem", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}, "name": "tf.__operators__.getitem", "inbound_nodes": [["input_1", 0, 0, {"slice_spec": {"class_name": "__tuple__", "items": [{"start": null, "stop": null, "step": null}, {"start": 0, "stop": 1, "step": null}]}}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["tf.math.subtract_1", 0, 0, {"y": 3.0521126069900983, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["tf.math.subtract_2", 0, 0, {"y": 3.0521126069900983, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_3", "inbound_nodes": [["tf.__operators__.getitem_3", 0, 0, {"y": -1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["tf.math.subtract_4", 0, 0, {"y": 0.19999999999999998, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract", "inbound_nodes": [["tf.__operators__.getitem", 0, 0, {"y": -1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_1", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_1", "inbound_nodes": [["tf.math.multiply_1", 0, 0, {"y": 2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_2", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_2", "inbound_nodes": [["tf.math.multiply_2", 0, 0, {"y": 2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["tf.math.subtract_3", 0, 0, {"y": 3.0521126069900983, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_5", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_5", "inbound_nodes": [["tf.math.multiply_4", 0, 0, {"y": 2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["tf.math.subtract", 0, 0, {"y": 4.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_1", "inbound_nodes": [["_CONSTANT_VALUE", -1, 3.0791139602661133, {"y": ["tf.math.truediv_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_2", "inbound_nodes": [["_CONSTANT_VALUE", -1, 3.0791139602661133, {"y": ["tf.math.truediv_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_3", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_3", "inbound_nodes": [["tf.math.multiply_3", 0, 0, {"y": 2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 0.10000000149011612, {"y": ["tf.math.truediv_5", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv", "inbound_nodes": [["tf.math.multiply", 0, 0, {"y": 2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp", "inbound_nodes": [["tf.__operators__.add_1", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp_1", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp_1", "inbound_nodes": [["tf.__operators__.add_2", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_3", "inbound_nodes": [["_CONSTANT_VALUE", -1, 3.0791139602661133, {"y": ["tf.math.truediv_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.negative", "trainable": true, "dtype": "float32", "function": "math.negative"}, "name": "tf.math.negative", "inbound_nodes": [["tf.__operators__.add_6", 0, 0, {"name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add", "inbound_nodes": [["_CONSTANT_VALUE", -1, 0.0, {"y": ["tf.math.truediv", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_4", "inbound_nodes": [["tf.math.exp", 0, 0, {"y": ["tf.math.exp_1", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp_2", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp_2", "inbound_nodes": [["tf.__operators__.add_3", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["tf.math.negative", 0, 0, {"y": ["tf.__operators__.add", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_5", "inbound_nodes": [["tf.__operators__.add_4", 0, 0, {"y": ["tf.math.exp_2", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp_3", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp_3", "inbound_nodes": [["tf.math.multiply_5", 0, 0, {}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_4", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_4", "inbound_nodes": [["tf.__operators__.add_5", 0, 0, {"y": 3.0, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_6", "inbound_nodes": [["_CONSTANT_VALUE", -1, 100.0, {"y": ["tf.math.exp_3", 0, 0], "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_5", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_5", "inbound_nodes": [["tf.math.truediv_4", 0, 0, {"y": ["tf.math.multiply_6", 0, 0], "name": null}]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 0.10000000149011612, {"y": ["tf.math.subtract_5", 0, 0], "name": null}]]}, {"class_name": "HighwayLayer", "config": {"layer was saved without config": true}, "name": "highway_layer", "inbound_nodes": [{"previous_layer": ["dense", 0, 0, {}], "original_variable": ["input_1", 0, 0, {}]}]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.exp_4", "trainable": true, "dtype": "float32", "function": "math.exp"}, "name": "tf.math.exp_4", "inbound_nodes": [["tf.math.multiply_7", 0, 0, {}]]}, {"class_name": "HighwayLayer", "config": {"layer was saved without config": true}, "name": "highway_layer_1", "inbound_nodes": [{"previous_layer": ["highway_layer", 0, 0, {}], "original_variable": ["input_1", 0, 0, {}]}]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_7", "inbound_nodes": [["_CONSTANT_VALUE", -1, 1.0, {"y": ["tf.math.exp_4", 0, 0], "name": null}]]}, {"class_name": "HighwayLayer", "config": {"layer was saved without config": true}, "name": "highway_layer_2", "inbound_nodes": [{"previous_layer": ["highway_layer_1", 0, 0, {}], "original_variable": ["input_1", 0, 0, {}]}]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.log", "trainable": true, "dtype": "float32", "function": "math.log"}, "name": "tf.math.log", "inbound_nodes": [["tf.__operators__.add_7", 0, 0, {"name": null}]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["highway_layer_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.truediv_6", "trainable": true, "dtype": "float32", "function": "math.truediv"}, "name": "tf.math.truediv_6", "inbound_nodes": [["tf.math.log", 0, 0, {"y": 0.1, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}, "name": "tf.__operators__.add_8", "inbound_nodes": [["dense_1", 0, 0, {"y": ["tf.math.truediv_6", 0, 0], "name": null}]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["tf.__operators__.add_8", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 11]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 11]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "DPDEModel"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 11]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
8	keras_api"?
_tf_keras_layer?{"class_name": "SlicingOpLambda", "name": "tf.__operators__.getitem_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.getitem_1", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}}
?
9	keras_api"?
_tf_keras_layer?{"class_name": "SlicingOpLambda", "name": "tf.__operators__.getitem_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.getitem_2", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}}
?
:	keras_api"?
_tf_keras_layer?{"class_name": "SlicingOpLambda", "name": "tf.__operators__.getitem_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.getitem_4", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}}
?
;	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?
<	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_2", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?
=	keras_api"?
_tf_keras_layer?{"class_name": "SlicingOpLambda", "name": "tf.__operators__.getitem_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.getitem_3", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}}
?
>	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_4", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "SlicingOpLambda", "name": "tf.__operators__.getitem", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.getitem", "trainable": true, "dtype": "float32", "function": "__operators__.getitem"}}
?
@	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
A	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
B	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_3", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?
C	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
D	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?
E	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv_1", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
F	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv_2", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
G	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv_5", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
I	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
J	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_1", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
K	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_2", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
L	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv_3", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
M	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_6", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
N	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
O	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.exp", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.exp", "trainable": true, "dtype": "float32", "function": "math.exp"}}
?
P	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.exp_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.exp_1", "trainable": true, "dtype": "float32", "function": "math.exp"}}
?
Q	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_3", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
R	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.negative", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.negative", "trainable": true, "dtype": "float32", "function": "math.negative"}}
?
S	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
T	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_4", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
U	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.exp_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.exp_2", "trainable": true, "dtype": "float32", "function": "math.exp"}}
?
V	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
W	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_5", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
X	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.exp_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.exp_3", "trainable": true, "dtype": "float32", "function": "math.exp"}}
?
Y	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv_4", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
Z	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_6", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
[	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.subtract_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.subtract_5", "trainable": true, "dtype": "float32", "function": "math.subtract"}}
?

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 11}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11]}}
?
b	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_7", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
cUz
dUg
eUr
fUh
gWz
hWg
iWr
jWh
kbz
lbg
mbr
nbh
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "HighwayLayer", "name": "highway_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
s	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.exp_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.exp_4", "trainable": true, "dtype": "float32", "function": "math.exp"}}
?
tUz
uUg
vUr
wUh
xWz
yWg
zWr
{Wh
|bz
}bg
~br
bh
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "HighwayLayer", "name": "highway_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_7", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
?Uz
?Ug
?Ur
?Uh
?Wz
?Wg
?Wr
?Wh
?bz
?bg
?br
?bh
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "HighwayLayer", "name": "highway_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.log", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.log", "trainable": true, "dtype": "float32", "function": "math.log"}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.truediv_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.truediv_6", "trainable": true, "dtype": "float32", "function": "math.truediv"}}
?
?	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.__operators__.add_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.__operators__.add_8", "trainable": true, "dtype": "float32", "function": "__operators__.add"}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate\m?]m?cm?dm?em?fm?gm?hm?im?jm?km?lm?mm?nm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?\v?]v?cv?dv?ev?fv?gv?hv?iv?jv?kv?lv?mv?nv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
?
\0
]1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\0
]1
c2
d3
e4
f5
g6
h7
i8
j9
k10
l11
m12
n13
t14
u15
v16
w17
x18
y19
z20
{21
|22
}23
~24
25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39"
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
3trainable_variables
?non_trainable_variables
4regularization_losses
5	variables
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
:Z2dense/kernel
:Z2
dense/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
^	variables
?non_trainable_variables
_trainable_variables
?layer_metrics
`regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:Z2Uz
:Z2Ug
:Z2Ur
:Z2Uh
:ZZ2Wz
:ZZ2Wg
:ZZ2Wr
:ZZ2Wh
:Z2bz
:Z2bg
:Z2br
:Z2bh
v
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11"
trackable_list_wrapper
v
c0
d1
e2
f3
g4
h5
i6
j7
k8
l9
m10
n11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
o	variables
?non_trainable_variables
ptrainable_variables
?layer_metrics
qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:Z2Uz
:Z2Ug
:Z2Ur
:Z2Uh
:ZZ2Wz
:ZZ2Wg
:ZZ2Wr
:ZZ2Wh
:Z2bz
:Z2bg
:Z2br
:Z2bh
v
t0
u1
v2
w3
x4
y5
z6
{7
|8
}9
~10
11"
trackable_list_wrapper
v
t0
u1
v2
w3
x4
y5
z6
{7
|8
}9
~10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:Z2Uz
:Z2Ug
:Z2Ur
:Z2Uh
:ZZ2Wz
:ZZ2Wg
:ZZ2Wr
:ZZ2Wh
:Z2bz
:Z2bg
:Z2br
:Z2bh
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
 :Z2dense_1/kernel
:2dense_1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?trainable_variables
?layer_metrics
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
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
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047"
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
#:!Z2Adam/dense/kernel/m
:Z2Adam/dense/bias/m
:Z2	Adam/Uz/m
:Z2	Adam/Ug/m
:Z2	Adam/Ur/m
:Z2	Adam/Uh/m
:ZZ2	Adam/Wz/m
:ZZ2	Adam/Wg/m
:ZZ2	Adam/Wr/m
:ZZ2	Adam/Wh/m
:Z2	Adam/bz/m
:Z2	Adam/bg/m
:Z2	Adam/br/m
:Z2	Adam/bh/m
:Z2	Adam/Uz/m
:Z2	Adam/Ug/m
:Z2	Adam/Ur/m
:Z2	Adam/Uh/m
:ZZ2	Adam/Wz/m
:ZZ2	Adam/Wg/m
:ZZ2	Adam/Wr/m
:ZZ2	Adam/Wh/m
:Z2	Adam/bz/m
:Z2	Adam/bg/m
:Z2	Adam/br/m
:Z2	Adam/bh/m
:Z2	Adam/Uz/m
:Z2	Adam/Ug/m
:Z2	Adam/Ur/m
:Z2	Adam/Uh/m
:ZZ2	Adam/Wz/m
:ZZ2	Adam/Wg/m
:ZZ2	Adam/Wr/m
:ZZ2	Adam/Wh/m
:Z2	Adam/bz/m
:Z2	Adam/bg/m
:Z2	Adam/br/m
:Z2	Adam/bh/m
%:#Z2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
#:!Z2Adam/dense/kernel/v
:Z2Adam/dense/bias/v
:Z2	Adam/Uz/v
:Z2	Adam/Ug/v
:Z2	Adam/Ur/v
:Z2	Adam/Uh/v
:ZZ2	Adam/Wz/v
:ZZ2	Adam/Wg/v
:ZZ2	Adam/Wr/v
:ZZ2	Adam/Wh/v
:Z2	Adam/bz/v
:Z2	Adam/bg/v
:Z2	Adam/br/v
:Z2	Adam/bh/v
:Z2	Adam/Uz/v
:Z2	Adam/Ug/v
:Z2	Adam/Ur/v
:Z2	Adam/Uh/v
:ZZ2	Adam/Wz/v
:ZZ2	Adam/Wg/v
:ZZ2	Adam/Wr/v
:ZZ2	Adam/Wh/v
:Z2	Adam/bz/v
:Z2	Adam/bg/v
:Z2	Adam/br/v
:Z2	Adam/bh/v
:Z2	Adam/Uz/v
:Z2	Adam/Ug/v
:Z2	Adam/Ur/v
:Z2	Adam/Uh/v
:ZZ2	Adam/Wz/v
:ZZ2	Adam/Wg/v
:ZZ2	Adam/Wr/v
:ZZ2	Adam/Wh/v
:Z2	Adam/bz/v
:Z2	Adam/bg/v
:Z2	Adam/br/v
:Z2	Adam/bh/v
%:#Z2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
?2?
!__inference__wrapped_model_101233?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
F__inference_dpde_model_layer_call_and_return_conditional_losses_101886
F__inference_dpde_model_layer_call_and_return_conditional_losses_101715
F__inference_dpde_model_layer_call_and_return_conditional_losses_103052
F__inference_dpde_model_layer_call_and_return_conditional_losses_102797?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dpde_model_layer_call_fn_102431
+__inference_dpde_model_layer_call_fn_102159
+__inference_dpde_model_layer_call_fn_103254
+__inference_dpde_model_layer_call_fn_103153?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_103265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_103274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_highway_layer_layer_call_and_return_conditional_losses_103331?
???
FullArgSpec%
args?
jself
jinput_combined
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_highway_layer_layer_call_fn_103361?
???
FullArgSpec%
args?
jself
jinput_combined
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_103418?
???
FullArgSpec%
args?
jself
jinput_combined
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_highway_layer_1_layer_call_fn_103448?
???
FullArgSpec%
args?
jself
jinput_combined
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_103505?
???
FullArgSpec%
args?
jself
jinput_combined
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_highway_layer_2_layer_call_fn_103535?
???
FullArgSpec%
args?
jself
jinput_combined
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_103545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_103554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_102542input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7?
!__inference__wrapped_model_101233?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????0?-
&?#
!?
input_1?????????
? "O?L
J
tf.__operators__.add_80?-
tf.__operators__.add_8??????????
C__inference_dense_1_layer_call_and_return_conditional_losses_103545^??/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????
? }
(__inference_dense_1_layer_call_fn_103554Q??/?,
%?"
 ?
inputs?????????Z
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_103265\\]/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????Z
? y
&__inference_dense_layer_call_fn_103274O\]/?,
%?"
 ?
inputs?????????
? "??????????Z?
F__inference_dpde_model_layer_call_and_return_conditional_losses_101715?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_dpde_model_layer_call_and_return_conditional_losses_101886?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_dpde_model_layer_call_and_return_conditional_losses_102797?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_dpde_model_layer_call_and_return_conditional_losses_103052?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
+__inference_dpde_model_layer_call_fn_102159?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????8?5
.?+
!?
input_1?????????
p

 
? "???????????
+__inference_dpde_model_layer_call_fn_102431?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????8?5
.?+
!?
input_1?????????
p 

 
? "???????????
+__inference_dpde_model_layer_call_fn_103153?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????7?4
-?*
 ?
inputs?????????
p

 
? "???????????
+__inference_dpde_model_layer_call_fn_103254?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
K__inference_highway_layer_1_layer_call_and_return_conditional_losses_103418?tx|uy}vz~w{???
???
???
O
original_variable:?7
 input_combined/original_variable?????????
I
previous_layer7?4
input_combined/previous_layer?????????Z
? "%?"
?
0?????????Z
? ?
0__inference_highway_layer_1_layer_call_fn_103448?tx|uy}vz~w{???
???
???
O
original_variable:?7
 input_combined/original_variable?????????
I
previous_layer7?4
input_combined/previous_layer?????????Z
? "??????????Z?
K__inference_highway_layer_2_layer_call_and_return_conditional_losses_103505????????????????
???
???
O
original_variable:?7
 input_combined/original_variable?????????
I
previous_layer7?4
input_combined/previous_layer?????????Z
? "%?"
?
0?????????Z
? ?
0__inference_highway_layer_2_layer_call_fn_103535????????????????
???
???
O
original_variable:?7
 input_combined/original_variable?????????
I
previous_layer7?4
input_combined/previous_layer?????????Z
? "??????????Z?
I__inference_highway_layer_layer_call_and_return_conditional_losses_103331?cgkdhleimfjn???
???
???
O
original_variable:?7
 input_combined/original_variable?????????
I
previous_layer7?4
input_combined/previous_layer?????????Z
? "%?"
?
0?????????Z
? ?
.__inference_highway_layer_layer_call_fn_103361?cgkdhleimfjn???
???
???
O
original_variable:?7
 input_combined/original_variable?????????
I
previous_layer7?4
input_combined/previous_layer?????????Z
? "??????????Z?
$__inference_signature_wrapper_102542?F???????\]cgkdhleimfjn?tx|uy}vz~w{??????????????;?8
? 
1?.
,
input_1!?
input_1?????????"O?L
J
tf.__operators__.add_80?-
tf.__operators__.add_8?????????