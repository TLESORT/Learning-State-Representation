
require "optim"
dofile("functions.lua")
dofile("Get_Images_Set.lua")
dofile("printing.lua")


function show_figure(list_out1, Name, Variable_Name)
	local Variable_Name=Variable_Name or 'out1'
	-- log results to files
	local accLogger = optim.Logger(Name)


	for i=1, (#list_out1[1])[1] do
	-- update logger
		accLogger:add{[Variable_Name.."-DIM-1"] = list_out1[1][i],
				[Variable_Name.."-DIM-2"] = list_out1[2][i],
				[Variable_Name.."-DIM-3"] = list_out1[3][i]}
	end
	-- plot logger
	accLogger:style{[Variable_Name.."-DIM-1"] = '-',
			[Variable_Name.."-DIM-2"] = '-',
			[Variable_Name.."-DIM-3"] = '-'}
	
	accLogger.showPlot = true
	accLogger:plot()
end

function show_MI(list_out1,ref, Name, Variable_Name)
	local Variable_Name=Variable_Name or 'Mutual information'
	-- log results to files
	local accLogger = optim.Logger(Name)


	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name] = list_out1[i],["ref"]=ref}
	end
	-- plot logger
	accLogger:style{[Variable_Name] = '-',["ref"]="-"}
	
	accLogger.showPlot = true
	accLogger:plot()
end

list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
indice_test=4
txt_test=list_txt_state[indice_test]
truth=getTruth(txt_test)
Truth=torch.Tensor(3,#truth)
 for i=1, #truth do
for j=1, 3 do
Truth[j][i]=truth[i][j]
end
end
ref=mutual_information(Truth, Truth)
print(ref)
mean_1=Truth[1]:mean()
mean_2=Truth[2]:mean()
mean_3=Truth[3]:mean()
std_1=Truth[1]:std()
std_2=Truth[2]:std()
std_3=Truth[3]:std()
noise=torch.rand(3,#truth)
noise[1]=noise[1]-noise[1]:mean()+mean_1
noise[2]=noise[2]-noise[2]:mean()+mean_2
noise[3]=noise[3]-noise[3]:mean()+mean_3
noise[1]=(noise[1]/noise[1]:std())*std_1
noise[2]=(noise[2]/noise[2]:std())*std_2
noise[3]=(noise[3]/noise[3]:std())*std_3

fact=1
TN=(Truth+noise/fact)*fact/(fact+1)
print(mutual_information(Truth, TN))
ComputeCorrelation(Truth, TN,3,"correlation")
name="./Test_Mi"
show_figure(Truth, name.."/TRUTH.log")
show_figure(TN, name.."/noise_Div"..fact..".log")


--[[

res={}
for i=1, 100 do
 	table.insert(res,mutual_information(Truth, (Truth+noise/i)))
end
show_MI(res,ref, name.."/evolution.log", Variable_Name)
--]]

