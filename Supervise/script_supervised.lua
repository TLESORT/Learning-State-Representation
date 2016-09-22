require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'

require '../MSDC'
require '../functions.lua'
require '../printing.lua'
require "../Get_Images_Set"
require '../priors'

function Training(Models,Mode,batch,Label,criterion,coef,LR)
	local LR=LR or 0.001
	local mom=0.0--9
        local coefL2=0,0


		 -- just in case:
	collectgarbage()
	batch=batch:cuda()
	Label=Label:cuda()

	 -- reset gradients
	State1=Model:forward(batch)
	criterion=criterion:cuda()
	output=criterion:forward({State1, Label})
	Model:zeroGradParameters()
	GradOutputs=criterion:backward({State1, Label})
	Model:backward(batch,GradOutputs[1])

	Model:updateParameters(LR)
	--optimState={learningRate=LR}
	--parameters, loss=optim.adagrad(feval, parameters, optimState)

	return loss
end

function load_Part_supervised(list,txt_state,im_lenght,im_height,nb_part,part)
	local x=2
	local y=3
	local z=4
	local train=false
	local Data={Images={},Labels={}}
	local list_lenght = torch.floor(#list/nb_part)
	local start=list_lenght*part +1
	local tensor, label=tensorFromTxt(txt_state)
	for i=start, start+list_lenght do
		local Label=torch.Tensor(3)
		table.insert(Data.Images,getImage(list[i],im_lenght,im_height,train))	
		Label[1]=tensor[i][x]
		Label[2]=tensor[i][y]
		Label[3]=tensor[i][z]
		table.insert(Data.Labels,Label*10)
	end 
	return Data
end
function Print_Supervised(Model,imgs, name, Log_Folder, truth)

	local list_out1={}

	for i=1, #imgs do --#imgs do
		image1=imgs[i]
		Data1=torch.Tensor(1,3,200,200)
		Data1[1]=image1
		Model:forward(Data1:cuda())
		local State1= torch.Tensor(3)
		State1:copy(Model.output)
		table.insert(list_out1,State1)
	end
	ComputeCorrelation(truth,list_out1,3)
	show_figure(list_out1, Log_Folder..'state'..name..'.log', 100)
end

function train_Epoch(Models,Log_Folder,LR)
	local BatchSize=1
	local nbEpoch=100
	
	local name='Save'..day
	local name_save=Log_Folder..name..'.t7'
	local criterion=nn.MSDCriterion()
	local list_errors={}

	local list_test=images_Paths(list_folders_images[nbList])
	local txt_test=list_txt_state[nbList]
	Data_test=load_Part_supervised(list_test,txt_test,image_width,image_height,10,0)
	imgs_test=Data_test.Images
	local truth=Data_test.Labels
	show_figure(truth, Log_Folder..'The_Truth.Log',100)

	Print_Supervised(Models, imgs_test,"First_Test",Log_Folder,truth)
			
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(nbList..' : sequences')
		indice1=torch.random(1,nbList-1)
		txt_state=list_txt_state[indice1]

		local nb_part=5
		local part=torch.random(0,nb_part-1)
		local list=images_Paths(list_folders_images[indice1])

		Data=load_Part_supervised(list,txt_state,image_width,image_height,nb_part,part)
		local nb_passage=10
		for j=1, nb_passage do
			for i=1, #Data.Images do
				batch=torch.Tensor(1,3, 200, 200)
				batch[1]=Data.Images[i]
				Training(Models,Mode,batch,Data.Labels[i],criterion,coef,LR)
			end
				xlua.progress(j, nb_passage)
		end
		
		save_model(Model,name_save)
		Print_Supervised(Model,imgs_test,name..epoch.."_Test",Log_Folder,truth)
	end

end


day="22-09"
local UseSecondGPU= true
local LR=0.005
local Dimension=3

local Log_Folder='./Log/'..day..'/'

name_load='./Log/Save/'..day..'.t7'

list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
local model_file='../models/topUniqueFM_Deeper2'


image_width=200
image_height=200

if UseSecondGPU then
	cutorch.setDevice(2) 
end

nbList= #list_folders_images
torch.manualSeed(123)

require(model_file)
Model=getModel(Dimension)	
Model=Model:cuda()
parameters,gradParameters = Model:getParameters()
print("Test actuel : "..Log_Folder)
train_Epoch(Model,Log_Folder,LR)

imgs={} --memory is free!!!!!
