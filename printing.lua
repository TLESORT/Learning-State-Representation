---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_performance(Models,imgs,txt,txt_reward, name, Log_Folder, truth)

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp=0
	local Rep=0
	local Prop=0
	local Caus=0
	local Model=Models.Model1

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

	-- biased estimation of test loss
	local nb_sample=100
--[[
	for i=1, nb_sample do
		Prop_batch=getRandomBatch(imgs, txt,txt_reward, 1, 200, 200, 'Prop', use_simulate_images)
		Temp_batch=getRandomBatch(imgs, txt,txt_reward, 1, 200, 200, 'Temp', use_simulate_images)
		Caus_batch=getRandomBatch(imgs, txt,txt_reward, 1, 200, 200, 'Caus', use_simulate_images)
		
		Temp=Temp+doStuff_temp(Models,TEMP_criterion, Temp_batch)
		Prop=Prop+doStuff_Prop(Models,PROP_criterion,Prop_batch)	
		Caus=Caus+doStuff_Caus(Models,CAUS_criterion,Caus_batch)
		Rep=Rep+doStuff_Rep(Models,REP_criterion,Prop_batch)
	end--]]
	ComputeCorrelation(truth,list_out1,3)
	show_figure(list_out1, Log_Folder..'state'..name..'.log', 1000)
	return Temp/nb_sample,Prop/nb_sample, Rep/nb_sample, Caus/nb_sample, list_out1
end


function ComputeCorrelation(truth,output,dimension)
	Truth=torch.Tensor(dimension,#truth)
	Output=torch.Tensor(dimension,#output)
	for i=1, #truth do
		for j=1, dimension do
			Truth[j][i]=truth[i][j]
			Output[j][i]=output[i][j]
		end
	end
	corr=torch.Tensor(dimension,dimension)

	for i=1,dimension do
		for j=1, dimension do
			corr[i][j]=torch.cmul((Truth[i]-Truth[i]:mean()),(Output[j]-Output[j]:mean())):mean()
			corr[i][j]=corr[i][j]/(Truth[i]:std()*Output[j]:std())
		end
	end

	print("Coorelation")
	print(corr)
	for i=1,dimension do
		for j=1, dimension do
			corr[i][j]=torch.cmul((Output[i]-Output[i]:mean()),(Output[j]-Output[j]:mean())):mean()
			corr[i][j]=corr[i][j]/(Output[i]:std()*Output[j]:std())
		end
	end

	print("InterCoorelation output")
	print(corr)
	for i=1,dimension do
		for j=1, dimension do
			corr[i][j]=torch.cmul((Truth[i]-Truth[i]:mean()),(Truth[j]-Truth[j]:mean())):mean()
			corr[i][j]=corr[i][j]/(Truth[i]:std()*Truth[j]:std())
		end
	end

	print("Coorelation truth")
	print(corr)

end

---------------------------------------------------------------------------------------
-- Function : Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list,Log_Folder)

	local scale= 1000
	local Name = Log_Folder..'Grad.log'
	local accLogger = optim.Logger(Name)

	for i=1, #Temp_grad_list do
	-- update logger
		accLogger:add{['Temp_Grad*'..scale] = Temp_grad_list[i]*scale,
				['Prop_Grad*'..scale] = Prop_grad_list[i]*scale,
				['Rep_Grad*'..scale] = Rep_grad_list[i]*scale,
				['Caus_Grad*'..scale] = Caus_grad_list[i]*scale}
	end
	-- plot logger
	accLogger:style{['Temp_Grad*'..scale] = '-',
			['Prop_Grad*'..scale] = '-',
			['Rep_Grad*'..scale] = '-',
			['Caus_Grad*'..scale] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : show_loss(list_train, list_test, Name , scale)
-- Input (list_train): list of the train loss
-- Input (list_test): list of the test loss
-- Input (Name): Name of the file
-- Input (scale): multiplicator factor needed because for optim.logger 1.1=1 but 11~=10
---------------------------------------------------------------------------------------
function show_loss(list_train, list_test, Name , scale)

	local scale=scale or 1000
	-- log results to files
	local accLogger = optim.Logger(Name)

	for i=1, #list_train do
	-- update logger
		accLogger:add{['train*'..scale] = list_train[i]*scale,['test*'..scale] = list_test[i]*scale}
	end
	-- plot logger
	accLogger:style{['train*'..scale] = '-',['test*'..scale] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : show_figure(list_out1, Name , scale)
-- Input (list_out1): list of the estimate state
-- Input (Name) : Name of the file
-- Input (scale) : multiplicator factor needed because for optim.logger 1.1=1 but 11~=10
---------------------------------------------------------------------------------------
function show_figure(list_out1, Name , scale, Variable_Name)

	Variable_Name=Variable_Name or 'out1'

	local scale=scale or 1000
	-- log results to files
	accLogger = optim.Logger(Name)

	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name.."-DIM-1"] = list_out1[i][1]*scale,
				[Variable_Name.."-DIM-2"] = list_out1[i][2]*scale,
				[Variable_Name.."-DIM-3"] = list_out1[i][3]*scale}
	end
	-- plot logger
	accLogger:style{[Variable_Name.."-DIM-1"] = '-',[Variable_Name.."-DIM-2"] = '-',[Variable_Name.."-DIM-3"] = '-'}
	
	accLogger.showPlot = false
	accLogger:plot()
end
