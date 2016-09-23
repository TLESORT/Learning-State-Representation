require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'

require 'MSDC'
require 'functions.lua'
require 'printing.lua'
require "Get_Images_Set"
require 'priors'

function Rico_Training(Models,Mode,Data1,Data2,criterion,coef,LR,BatchSize)
	local LR=LR or 0.001
	local mom=0.9
	local coefL2=0,0

	local batch=getRandomBatchFromSeparateList(Data1,Data2,BatchSize,Mode)

	      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
		 -- just in case:
		 collectgarbage()

		 -- get new parameters
		 if x ~= parameters then
		    parameters:copy(x)
		 end

		 -- reset gradients
		gradParameters:zero()
		if Mode=='Simpl' then print("Simpl")
		elseif Mode=='Temp' then loss,grad=doStuff_temp(Models,criterion, batch,coef)
		elseif Mode=='Prop' then loss,grad=doStuff_Prop(Models,criterion,batch,coef)
		elseif Mode=='Caus' then loss,grad=doStuff_Caus(Models,criterion,batch,coef)
		elseif Mode=='Rep' then loss,grad=doStuff_Rep(Models,criterion,batch,coef)
		else print("Wrong Mode")
		end
         	return loss,gradParameters
	end
        sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
	parameters, loss=optim.sgd(feval, parameters, sgdState)

--optimState={learningRate=LR}
--parameters, loss=optim.adagrad(feval, parameters, optimState)

	 -- loss[1] table of one value transformed in just a value
	 -- grad[1] we use just the first gradient to print the figure (there are 2 or 4 gradient normally)

	return loss[1], grad:mean()
end


function train_Epoch(Models,Prior_Used,Log_Folder,LR)
	local nbEpoch=100
	local NbBatch=100
	local BatchSize=2
	
	local name='Save'..day
	local name_save=Log_Folder..name..'.t7'

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp_loss_list, Prop_loss_list, Rep_loss_list, Caus_loss_list = {},{},{},{}
	local Temp_loss_list_test,Prop_loss_list_test,Rep_loss_list_test,Caus_loss_list_test = {},{},{},{}
	local Sum_loss_train, Sum_loss_test = {},{}
	local Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list = {},{},{},{}		
	local list_errors={}

	local Prop=Have_Todo(Prior_Used,'Prop')
	local Temp=Have_Todo(Prior_Used,'Temp')
	local Rep=Have_Todo(Prior_Used,'Rep')
	local Caus=Have_Todo(Prior_Used,'Caus')
print(Prop)
print(Temp)
print(Rep)
print(Caus)

	local coef_Temp=1
	local coef_Prop=1
	local coef_Rep=1
	local coef_Caus=1
	local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}


	local list_truth=images_Paths(list_folders_images[nbList])
	txt_test=list_txt_state[nbList]
	txt_reward_test=list_txt_button[nbList]
	Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,image_width,image_height,10,1,false)
	imgs_test=Data_test.images
	local truth=getTruth(txt_test)
	show_figure(truth, Log_Folder..'The_Truth.Log')
	Print_performance(Models, imgs_test,txt_test,txt_reward_test,"First_Test",Log_Folder,truth)

	--real_temp_loss,real_prop_loss,real_rep_loss, real_caus_loss=real_loss(txt_test)
	--print("temp loss : "..real_temp_loss)
	--print("prop loss : "..real_prop_loss[1])
	--print("rep loss : "..real_rep_loss[1])	
	--print("caus loss : "..real_caus_loss[1])
	--printParamInAFile(Log_Folder,coef_list, LR, "adagrad", BatchSize, nbEpoch, NbBatch)
			
	for epoch=1, nbEpoch do
		print('--------------Epoch : '..epoch..' ---------------')
		print(nbList..' : sequences')

		local Temp_loss=0
		local Prop_loss=0
		local Rep_loss=0
		local Caus_loss=0

		local Grad_Temp=0
		local Grad_Prop=0
		local Grad_Rep=0
		local Grad_Caus=0

		indice1=torch.random(1,nbList-1)
		repeat indice2=torch.random(1,nbList-1) until (indice1 ~= indice2)

		txt1=list_txt_action[indice1]
		txt2=list_txt_action[indice2]
		txt_reward1=list_txt_button[indice1]
		txt_reward2=list_txt_button[indice2]

		local nb_part=5
		local part1=torch.random(0,nb_part-1)
		local part2=torch.random(0,nb_part-1)
-- for debug
		list1=images_Paths(list_folders_images[indice1])
		list2=images_Paths(list_folders_images[indice2])
		Data1,ThereIsReward=load_Part_list(list1,txt1,txt_reward1,image_width,image_height,nb_part,part1,false)--without data augmentation
		Data2,ThereIsReward2=load_Part_list(list2,txt2,txt_reward2,image_width,image_height,nb_part,part2,false)--without data augmentation

	printParamInAFile(Log_Folder,coef_list, LR, "LR+mom", BatchSize, nbEpoch, NbBatch, model_file)

		for numBatch=1, NbBatch do
			if Temp then
				Loss,Grad=Rico_Training(Models,'Temp',Data1,Data2,TEMP_criterion, coef_Temp,LR,BatchSize)
				Grad_Temp=Grad_Temp+Grad
 				Temp_loss=Temp_loss+Loss
			end
			if Prop then
				Loss,Grad=Rico_Training(Models,'Prop',Data1,Data2, PROP_criterion, coef_Prop,LR,BatchSize)
				Grad_Prop=Grad_Prop+Grad
				Prop_loss=Prop_loss+Loss
			end
			if Rep then
				Loss,Grad=Rico_Training(Models,'Rep',Data1,Data2,REP_criterion, coef_Rep,LR,BatchSize)
				Grad_Rep=Grad_Rep+Grad
				Rep_loss=Rep_loss+Loss
			end
			if Caus and (ThereIsReward and ThereIsReward2) then 
				Loss,Grad=Rico_Training(Models, 'Caus',Data1,Data2, CAUS_criterion, coef_Caus,LR,BatchSize)
				Grad_Caus=Grad_Caus+Grad
				Caus_loss=Caus_loss+Loss
			end
			xlua.progress(numBatch, NbBatch)
		end
		
		save_model(Models.Model1,name_save)
		local id=name..epoch -- variable used to not mix several log files
		Temp_test,Prop_test,Rep_test,Caus_test, list_estimation=Print_performance(Models, imgs_test,txt_test,txt_reward_test,id.."_Test",Log_Folder,truth)

		table.insert(Temp_loss_list,Temp_loss/NbBatch)
		table.insert(Prop_loss_list,Prop_loss/NbBatch)
		table.insert(Rep_loss_list,Rep_loss/NbBatch)		
		table.insert(Caus_loss_list,Caus_loss/NbBatch)

		table.insert(Temp_loss_list_test,Temp_test)
		table.insert(Prop_loss_list_test,Prop_test)
		table.insert(Rep_loss_list_test,Rep_test)
		table.insert(Caus_loss_list_test,Caus_test)

		table.insert(Temp_grad_list,Grad_Temp)
		table.insert(Prop_grad_list,Grad_Prop/NbBatch)
		table.insert(Rep_grad_list,Grad_Rep/NbBatch)
		table.insert(Caus_grad_list,Grad_Caus/NbBatch)

		
		sum_train=(Temp_loss+Prop_loss+Rep_loss+Caus_loss)/NbBatch
		table.insert(Sum_loss_train,sum_train)
		table.insert(Sum_loss_test,Temp_test+Prop_test+Rep_test+Caus_test)

		show_loss(Temp_loss_list,Temp_loss_list_test, Log_Folder..'Temp_loss.log', 1000)
		show_loss(Prop_loss_list,Prop_loss_list_test, Log_Folder..'Prop_loss.log', 1000)
		show_loss(Rep_loss_list,Rep_loss_list_test, Log_Folder..'Rep_loss.log', 1000)
		show_loss(Caus_loss_list,Caus_loss_list_test, Log_Folder..'Caus_loss.log', 1000)
		show_loss(Sum_loss_train,Sum_loss_test, Log_Folder..'Sum_loss.log', 1000)
		Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list,Log_Folder)
	end

end


day="23-09"
local UseSecondGPU= false
local LR=0.001
local Dimension=3

Tests_Todo={{"Prop","Rep","Caus","Temp"}}

local Log_Folder='./Log/'..day..'/'


name_load='./Log/Save/'..day..'.t7'

list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
local reload=false
local TakeWeightFromAE=false
model_file='./models/topUniqueFM_Deeper2'


image_width=200
image_height=200

if UseSecondGPU then
	cutorch.setDevice(2) 
end

nbList= #list_folders_images

for nb_test=1, #Tests_Todo do

	torch.manualSeed(123)

	if reload then
		Model = torch.load('./Log/13_09_adagrad4_coef1/Everything/Save13_09_adagrad4_coef1.t7'):double()
	elseif TakeWeightFromAE then
		require './Autoencoder/noiseModule'
		require(model_file)
		Model=getModel(image_width,image_height)
		AE= torch.load('./Log/13_09_adagrad4_coef1/Everything/Save13_09_adagrad4_coef1.t7'):double()
		print('AE\n' .. AE:__tostring());
		Model=copy_weight(Model, AE)
	else
		require(model_file)
		Model=getModel(Dimension)	
	end
	Model=Model:cuda()
	parameters,gradParameters = Model:getParameters()
	Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
	Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
	Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

	Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}

	local Priors=Tests_Todo[nb_test]
	local Log_Folder=Get_Folder_Name(Log_Folder,Priors)
	print("Test actuel : "..Log_Folder)
	train_Epoch(Models,Priors,Log_Folder,LR)
end

imgs={} --memory is free!!!!!
