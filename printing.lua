---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_performance(Models,Data,txt,txt_reward, name, Log_Folder, truth)

	local imgs=Data.images
	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()

	local Temp,Rep,Prop,Caus=0,0,0,0
	local Model=Models.Model1

	local list_out1={}

	for i=1, #imgs do --#imgs do
		image1=imgs[i]
		Data1=torch.Tensor(2,3,200,200)
		Data1[1]=image1
		Data1[2]=image1
		Model:forward(Data1:cuda())
		local State1= torch.Tensor(3) 
		State1:copy(Model.output[1])
		table.insert(list_out1,State1)
	end

	-- biased estimation of test loss
	local nb_sample=100

	for i=1, nb_sample do
		Prop_batch=getRandomBatch(Data, 2, "Prop")
		Temp_batch=getRandomBatch(Data, 2, "Temp")
		Caus_batch=getRandomBatch(Data, 2, "Caus")
		
		Temp=Temp+doStuff_temp(Models,TEMP_criterion, Temp_batch)
		Prop=Prop+doStuff_Prop(Models,PROP_criterion,Prop_batch)	
		Caus=Caus+doStuff_Caus(Models,CAUS_criterion,Caus_batch)
		Rep=Rep+doStuff_Rep(Models,REP_criterion,Prop_batch)
	end
	Correlation, mutual_info=print_correlation(truth,list_out1,3)
	show_figure(list_out1, Log_Folder..'state'..name..'.log',"Estimation",Data.Infos)
	return Temp/nb_sample,Prop/nb_sample, Rep/nb_sample, Caus/nb_sample, list_out1, mutual_info, Correlation
end


function print_correlation(truth,output,dimension)
	Truth=torch.Tensor(dimension,#truth)
	Output=torch.Tensor(dimension,#output)
	for i=1, #truth do
		for j=1, dimension do
			Truth[j][i]=truth[i][j]
			Output[j][i]=output[i][j]
		end
	end
	Correlation=ComputeCorrelation(Truth,Output,dimension,"Correlation")
	--ComputeCorrelation(Truth,Truth,dimension,"InterCorrelation Truth")
	--ComputeCorrelation(Output,Output,dimension,"InterCorrelation output")

--!! attention si les axes sont inversés la correlation 3D sera fausse, (?)
	--ComputeMax3DCorrelation(Truth,Output,dimension,"Max 3D correlation")
	
	mutual_info=mutual_information(Truth, Truth:float())	
	print("Mutual Info Référence")
	print(mutual_info)
--[[	print("test IM")
	test=torch.Tensor(Truth:size())
	test:copy(Truth)
	D1=test[1]:clone()
	D2=test[2]:clone()	
	D3=test[3]:clone()
	--test[1],test[2],test[3]=test[1]-test[2],test[2]+2*test[1],4*test[3]
	test[1],test[2],test[3]=D2-D1,D1+D2,4*D3
	mutual_info=mutual_information(Truth, test)
	print(mutual_info)--]]
	print("Mutual Info")
	mutual_info=mutual_information(Truth, Output)
	print(mutual_info)

	return Correlation, mutual_info
end

function ComputeCorrelation(Truth,Output,dimension,label)
	local corr=torch.Tensor(dimension,dimension)
	for i=1,dimension do
		for j=1, dimension do
			corr[i][j]=torch.cmul((Truth[i]-Truth[i]:mean()),(Output[j]-Output[j]:mean())):mean()
			corr[i][j]=corr[i][j]/(Truth[i]:std()*Output[j]:std())
		end
	end
	print(label)
	print(corr)
	return corr
end

function mutual_information(Real, Estimate)
	local real=torch.floor(Real:clone()*1000)/1000
	local estimate=torch.floor(Estimate:clone()*1000)/1000
	local division=5
	local eps=0--.000001

	local pas_x_real=(real[1]:max()-real[1]:min())/division
	local pas_y_real=(real[2]:max()-real[2]:min())/division
	local pas_z_real=(real[3]:max()-real[3]:min())/division

	local pas_x_estimate=(estimate[1]:max()-estimate[1]:min())/division
	local pas_y_estimate=(estimate[2]:max()-estimate[2]:min())/division
	local pas_z_estimate=(estimate[3]:max()-estimate[3]:min())/division


	local prob_real=torch.zeros(division,division,division)
	local prob_estimate=torch.zeros(division,division,division)
	local prob_both=torch.zeros(division,division,division,division,division,division)

	for i=1 , real[1]:size(1) do
		for j=1, division do
			if real[1][i]<=(j*pas_x_real+real[1]:min()+eps) and real[1][i]>=((j-1)*pas_x_real+real[1]:min()-eps)  then x_real=j end
			if real[2][i]<=(j*pas_y_real+real[2]:min()+eps) and real[2][i]>=((j-1)*pas_y_real+real[2]:min()-eps)  then y_real=j end
			if real[3][i]<=(j*pas_z_real+real[3]:min()+eps) and real[3][i]>=((j-1)*pas_z_real+real[3]:min()-eps)  then z_real=j end

			if estimate[1][i]<=(j*pas_x_estimate+estimate[1]:min()+eps) and estimate[1][i]>=((j-1)*pas_x_estimate+estimate[1]:min()-eps) then x_estimate=j end
			if estimate[2][i]<=(j*pas_y_estimate+estimate[2]:min()+eps) and estimate[2][i]>=((j-1)*pas_y_estimate+estimate[2]:min()-eps) then y_estimate=j end
			if estimate[3][i]<=(j*pas_z_estimate+estimate[3]:min()+eps) and estimate[3][i]>=((j-1)*pas_z_estimate+estimate[3]:min()-eps) then z_estimate=j end
		end
		prob_real[x_real][y_real][z_real]=prob_real[x_real][y_real][z_real]+1
		prob_estimate[x_estimate][y_estimate][z_estimate]=prob_estimate[x_estimate][y_estimate][z_estimate]+1
		prob_both[x_real][y_real][z_real][x_estimate][y_estimate][z_estimate]=prob_both[x_real][y_real][z_real][x_estimate][y_estimate][z_estimate]+1
	end

	prob_real=prob_real/real[1]:size(1)
	prob_estimate=prob_estimate/real[1]:size(1)
	prob_both=prob_both/real[1]:size(1)


	local mutual_info=0
	for x=1 , division do
	for y=1 , division do
	for z=1 , division do

	for x2=1 , division do
	for y2=1 , division do
	for z2=1 , division do
		if prob_real[x][y][z]*prob_estimate[x2][y2][z2]*prob_both[x][y][z][x2][y2][z2] ~= 0 then
			mutual_info=mutual_info+prob_both[x][y][z][x2][y2][z2]*math.log(prob_both[x][y][z][x2][y2][z2]/(prob_real[x][y][z]*prob_estimate[x2][y2][z2]))
		end
	end
	end
	end

	end
	end
	end
	return mutual_info

end

function ComputeMax3DCorrelation(Truth,Output,dimension,label)

-- FEW CORRECTION TODO
	local res=0
	local norm_truth=Truth-Truth:mean()
	local prod_std=(Truth:std()*Output:std())
	local mean_output=Output:mean()
	local signal=torch.Tensor(Output:size())
	for i=1,dimension do
		for j=1, dimension do
			for k=1, dimension do
				if k~=j and k~=i and i~=j then
					signal[1]:copy(Output[i])
					signal[2]:copy(Output[j])
					signal[3]:copy(Output[k])
					corr=torch.cmul(norm_truth,(signal-mean_output)):mean()
					corr=corr/prod_std
					if math.abs(corr)>res then res=math.abs(corr) end
				end
			end
		end
	end
	print(label)
	print(res)
end

---------------------------------------------------------------------------------------
-- Function : Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Print_Grad(Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list,Log_Folder)

	local Name = Log_Folder..'Grad.log'
	local accLogger = optim.Logger(Name)

	for i=1, #Temp_grad_list do
	-- update logger
		accLogger:add{['Temp_Grad'] = Temp_grad_list[i],
				['Prop_Grad'] = Prop_grad_list[i],
				['Rep_Grad'] = Rep_grad_list[i],
				['Caus_Grad'] = Caus_grad_list[i]}
	end
	-- plot logger
	accLogger:style{['Temp_Grad'] = '-',
			['Prop_Grad'] = '-',
			['Rep_Grad'] = '-',
			['Caus_Grad'] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : show_loss(list_train, list_test, Name)
-- Input (list_train): list of the train loss
-- Input (list_test): list of the test loss
-- Input (Name): Name of the file
---------------------------------------------------------------------------------------
function show_loss(list_train, list_test, Name )
	-- log results to files
	local accLogger = optim.Logger(Name)

	for i=1, #list_train do
	-- update logger
		accLogger:add{['train'] = list_train[i],['test'] = list_test[i]}
	end
	-- plot logger
	accLogger:style{['train'] = '-',['test'] = '-'}
	accLogger.showPlot = false
	accLogger:plot()
end

---------------------------------------------------------------------------------------
-- Function : show_figure(list_out1, Name)
-- Input (list_out1): list of the estimate state
-- Input (Name) : Name of the file
---------------------------------------------------------------------------------------
function show_figure(list_out1, Name, Variable_Name,Infos)
	assert(#list_out1==#Infos.reward,"error list are not same lenght")
	local Variable_Name=Variable_Name or 'out1'
	-- log results to files
	local accLogger = optim.Logger(Name)


	for i=1, #list_out1 do
	-- update logger
		reward=Infos.reward[i]
		accLogger:add{[Variable_Name.."-DIM-1"] = list_out1[i][1],
				[Variable_Name.."-DIM-2"] = list_out1[i][2],
				[Variable_Name.."-DIM-3"] = list_out1[i][3],
				["Reward"] = reward}
	end
	-- plot logger
	accLogger:style{[Variable_Name.."-DIM-1"] = '-',
			[Variable_Name.."-DIM-2"] = '-',
			[Variable_Name.."-DIM-3"] = '-',
			["Reward"] = '+'}
	
	accLogger.showPlot = false
	accLogger:plot()
end

function show_MI(list_MI,path)
	-- log results to files
	accLogger = optim.Logger(path)
	for i=1, #list_MI do
		accLogger:add{["Information Mutuelle"] = list_MI[i]}
	end
	-- plot logger
	accLogger:style{["Information Mutuelle"] = '-'}
	
	accLogger.showPlot = false
	accLogger:plot()
end

function printParamInAFile(path,coef_list, LR, optim, BatchSize, nbEpoch, NbBatch, model)
	local file=path.."info.txt"
	local f=io.open(file, "w")
	f:write("Coef Temp    : "..coef_list[1].."\n")
	f:write("Coef Prop    : "..coef_list[2].."\n")
	f:write("Coef Rep     : "..coef_list[3].."\n")
	f:write("Coef Caus    : "..coef_list[4].."\n")
	f:write("\n")
	f:write("Learning Rate: "..LR.."\n")
	f:write("Optimisation : "..optim.."\n")
	f:write("BatchSize    : "..BatchSize.."\n")
	f:write("Nb Epoch     : "..nbEpoch.."\n")
	f:write("Nb Batch     : "..NbBatch.."\n")
	f:write("Model name   : "..model.."\n")
	f:close()
end

function Print_Corr(corr,epoch,path)
	local file=path.."corr.txt"
	local f=io.open(file, "a")
	local str='Epoch : '..epoch..'\n'
	str=str..corr[1][1]..' , '..corr[1][2]..' , '..corr[1][3]..'\n'
	str=str..corr[2][1]..' , '..corr[2][2]..' , '..corr[2][3]..'\n'
	str=str..corr[3][1]..' , '..corr[3][2]..' , '..corr[3][3]..'\n'
	str=str..'----------------------------------------------------\n'
	f:write(str)
	f:close()
end
