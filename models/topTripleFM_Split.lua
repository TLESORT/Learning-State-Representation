
require 'nn'

-- network-------------------------------------------------------
function getModel(Dimension)
	nbFilter=32
	input = nn.Identity()()
	conv1 = nn.SpatialConvolution(3, nbFilter, 3, 3)(input)
	BN1   = nn.SpatialBatchNormalization(nbFilter)(conv1)
	relu1 = nn.ReLU()(BN1)

	conv2 = nn.SpatialConvolution(nbFilter, 2*nbFilter, 3, 3)(relu1)
	BN2   = nn.SpatialBatchNormalization(2*nbFilter)(conv2)
	relu2 = nn.ReLU()(BN2)	

	conv3 = nn.SpatialConvolution(2*nbFilter, 4*nbFilter, 3, 3)(relu2)
	BN3   = nn.SpatialBatchNormalization(4*nbFilter)(conv3)
	relu3 = nn.ReLU()(BN3)
	MP    = nn.SpatialMaxPooling(2,2,2,2)(relu3)

	conv4 = nn.SpatialConvolution(4*nbFilter, 8*nbFilter, 3, 3)(MP)
	BN4   = nn.SpatialBatchNormalization(8*nbFilter)(conv4)
	relu4 = nn.ReLU()(BN4)
	MP2   = nn.SpatialMaxPooling(2,2,2,2)(relu4)

	conv5_1 = nn.SpatialConvolution(8*nbFilter, 1, 1, 1)(MP2)
	conv5_2 = nn.SpatialConvolution(8*nbFilter, 1, 1, 1)(MP2)
	conv5_3 = nn.SpatialConvolution(8*nbFilter, 1, 1, 1)(MP2)
	BN5_1   = nn.SpatialBatchNormalization(1)(conv5_1)
	BN5_2   = nn.SpatialBatchNormalization(1)(conv5_2)
	BN5_3   = nn.SpatialBatchNormalization(1)(conv5_3)
	relu5_1 = nn.ReLU()(BN5_1)
	relu5_2 = nn.ReLU()(BN5_2)
	relu5_3 = nn.ReLU()(BN5_3)

	View_1=nn.View(1*47*47)(relu5_1)   
	View_2=nn.View(1*47*47)(relu5_2) 
	View_3=nn.View(1*47*47)(relu5_3)
              
	W_1=nn.ReLU()(nn.Linear(47*47, 100)(View_1))   
	W_2=nn.ReLU()(nn.Linear(47*47, 100)(View_2))  
	W_3=nn.ReLU()(nn.Linear(47*47, 100)(View_3))  

  	W2_1=nn.ReLU()(nn.Linear(100, 100)(W_1))   
	W2_2=nn.ReLU()(nn.Linear(100, 100)(W_2))  
	W2_3=nn.ReLU()(nn.Linear(100, 100)(W_3)) 

  	W3_1=nn.Linear(100, 1)(W2_1)  
	W3_2=nn.Linear(100, 1)(W2_2)  
	W3_3=nn.Linear(100, 1)(W2_3)

	
	out=nn.JoinTable(2)({W3_1,W3_2,W3_3})
	
	gmod = nn.gModule({input}, {out})

	-- Initiallisation : "Understanding the difficulty of training deep feedforward neural networks"
	local method = 'xavier'
	local gmod = require('weight-init')(gmod, method)
	--print('Timnet\n' .. Timnet:__tostring());
	return gmod
end
