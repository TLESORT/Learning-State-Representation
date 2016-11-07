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

function Training(Models,Mode,batch,label,criterion,coef,LR)
	local LR=LR or 0.001
	local mom=0.0--9
        local coefL2=0,0


		 -- just in case:
	collectgarbage()
	batch=batch:cuda()
	Label=torch.Tensor(2,3) --pb of batch sorry for bad programming issue
	Label[1]=label
	Label[2]=label
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

	return output
end


function show_figure_sup(list_out1, Name, Variable_Name)
	local Variable_Name=Variable_Name or 'out1'
	-- log results to files
	local accLogger = optim.Logger(Name)


	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name.."-DIM-1"] = list_out1[i][1],
				[Variable_Name.."-DIM-2"] = list_out1[i][2],
				[Variable_Name.."-DIM-3"] = list_out1[i][3]}
	end
	-- plot logger
	accLogger:style{[Variable_Name.."-DIM-1"] = '-',
			[Variable_Name.."-DIM-2"] = '-',
			[Variable_Name.."-DIM-3"] = '-'}
	
	accLogger.showPlot = false
	accLogger:plot()
end

function load_Part_supervised(list,txt_state,im_lenght,im_height,nb_part,part, coef_DA)
	local x=2
	local y=3
	local z=4
	local Data={Images={},Labels={}}
	local list_lenght = torch.floor(#list/nb_part)
	local start=list_lenght*part +1
	local tensor, label=tensorFromTxt(txt_state)
	for i=start, start+list_lenght do
		local Label=torch.Tensor(3)
		table.insert(Data.Images,getImage(list[i],im_lenght,im_height,coef_DA))	
		Label[1]=tensor[i][x]
		Label[2]=tensor[i][y]
		Label[3]=tensor[i][z]
		table.insert(Data.Labels,Label*10)
	end 
	return Data
end
function Print_Supervised(Model,Data, name, Log_Folder,criterion)

	local list_out1={}
	local sum_loss=0
	local loss=0
	criterion=criterion:cuda()

	for i=1, #Data.Images do --#imgs do
		image1=Data.Images[i]
		Label=Data.Labels[i]:cuda()
		Data1=torch.Tensor(2,3,200,200)
		Data1[1]=image1
		Data1[2]=image1
		Model:forward(Data1:cuda())
		loss=loss+criterion:forward({Model.output[1], Label})
		local State1= torch.Tensor(3)
		State1:copy(Model.output[1])
		table.insert(list_out1,State1)
	end
	Correlation, mutual_info=print_correlation(Data.Labels,list_out1,3)
	show_figure_sup(list_out1, Log_Folder..'state'..name..'.log')
	return loss/#Data.Images, mutual_info, Correlation
end

function train_Epoch(Models,Log_Folder,LR)
	local BatchSize=1
	local nbEpoch=1000
	local list_MI= {}
	local name='Save'..day
	local name_save=Log_Folder..name..'.t7'
	local criterion=nn.MSDCriterion()
	local list_errors={}
indice_test=4 --nbList
nb_part=50
part_test=1
	local list_test=images_Paths(list_folders_images[indice_test])
	local txt_test=list_txt_state[indice_test]
	Data_test=load_Part_supervised(list_test,txt_test,image_width,image_height,nb_part,part_test,0)
	local sum_loss=0
	local Loss_Train={}
	local Loss_Valid={}
	show_figure_sup(Data_test.Labels, Log_Folder..'The_Truth.Log')

	Print_Supervised(Models, Data_test,"First_Test",Log_Folder,criterion)
			
	for epoch=1, nbEpoch do
		sum_loss=0
		print('--------------Epoch : '..epoch..' ---------------')
		print(nbList..' : sequences')
		indice1=torch.random(1,nbList-1)
indice1=4
		txt_state=list_txt_state[indice1]

		local nb_part=50
		local part=torch.random(2,nb_part-1)-- part 0 contain void images, 1 is for test
		local list=images_Paths(list_folders_images[indice1])

		Data=load_Part_supervised(list,txt_state,image_width,image_height,nb_part,part,0.1)
		local nb_passage=10
		for j=1, nb_passage do
			for i=1, #Data.Images do
				batch=torch.Tensor(2,3, 200, 200)
				batch[1]=Data.Images[i]
				batch[2]=Data.Images[i]
				loss=Training(Models,Mode,batch,Data.Labels[i],criterion,coef,LR)
				sum_loss=sum_loss+loss
			end
				xlua.progress(j, nb_passage)
		end

		save_model(Model,name_save)
		loss_test, mutual_info, corr=Print_Supervised(Model,Data_test,name..epoch.."_Test",Log_Folder,criterion)

		table.insert(Loss_Train,sum_loss/(#Data.Images*nb_passage))
		table.insert(Loss_Valid,loss_test)
		show_loss(Loss_Train,Loss_Valid, Log_Folder..'Mean_loss.log')

		table.insert(list_MI,mutual_info)
		show_MI(list_MI, Log_Folder..'Mutuelle_Info.log')
		Print_Corr(corr,epoch,Log_Folder)
	end

end


day="21-10"
local UseSecondGPU= false
local LR=0.005
local Dimension=3

local Log_Folder='./Log/'..day..'/'

name_load='./Log/Save/'..day..'.t7'

list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
local model_file='../models/topTripleFM_Split'


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
