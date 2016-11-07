---------------------------------------------------------------------------------------
-- Function :save_model(model,path) 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function save_model(model,path)
	print("Saved at : "..path)
	model:cuda()
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone():float()
	lightModel:clearState()
	torch.save(path,lightModel)
end



---------------------------------------------------------------------------------------
-- Function : preprocessing(im, lenght, width, SpacialNormalization)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function preprocessing(im, lenght, width,coef_DA)

		-- Name channels for convenience
	local channels = {'y','u','v'}
	local mean = {}
	local std = {}
	data = torch.Tensor( 3, im:size(2), im:size(3))
	data:copy(im)
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = data[i]:mean()
	   std[i] = data[{i,{},{}}]:std()
	   data[{i,{},{}}]:add(-mean[i])
	   data[{i,{},{}}]:div(std[i])
	end

	

--[[	local neighborhood = image.gaussian1D(5) -- 5 for face detector training
	-- Define our local normalization operator
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-4)
 
 
	-- Normalize all channels locally:
		-- Normalize all channels locally:
	for c in ipairs(channels) do
		for c in ipairs(channels) do
	      	      data[{{c},{},{} }] = normalization:forward(data[{{c},{},{} }])
		      data[{{c},{},{} }] = normalization:forward(data[{{c},{},{} }])
		end
 	end--]]
	if coef_DA ~=0 then data=dataAugmentation(data, lenght, width,coef_DA) end

	return data
end

local function gamma(im)
	local Gamma= torch.Tensor(3,3)
	local channels = {'y','u','v'}
	local mean = {}
	local std = {}
	for i,channel in ipairs(channels) do
		
		for j,channel in ipairs(channels) do
	   		if i==j then Gamma[i][i] = im[{i,{},{}}]:var()
			else
				chan_i=im[{i,{},{}}]-im[{i,{},{}}]:mean()
				chan_j=im[{j,{},{}}]-im[{j,{},{}}]:mean()
				Gamma[i][j]=(chan_i:t()*chan_j):mean()
			end
		end
	end

	return Gamma
end

local function transformation(im, v,e, fact)
	local transfo=torch.Tensor(3,200,200)
	local Gamma=torch.mv(v,e)
	for i=1, 3 do
		transfo[i]=im[i]+Gamma[i]*fact
	end
 return transfo
end


function loi_normal(x,y,center_x,center_y,std_x,std_y)
 return math.exp(-(x-center_x)^2/(2*std_x^2))*math.exp(-(y-center_y)^2/(2*std_y^2))
end
---------------------------------------------------------------------------------------
-- Function : dataAugmentation(im, lenght, width)
-- Input ():
-- Output ():
-- goal : By using data augmentation we want or network to be more resistant to no task relevant perturbations like luminosity variation or noise
---------------------------------------------------------------------------------------
function dataAugmentation(im, lenght, width,coef_DA)
	local channels = {'y','u','v'}	
	
	gam=gamma(im)
	e, V = torch.eig(gam,'V')
	factors=torch.randn(3)*0.1
	for i=1,3 do e:select(2, 1)[i]=e:select(2, 1)[i]*factors[i] end
	im=transformation(im, V,e:select(2, 1),coef_DA)
	noise=torch.rand(3,lenght,width)
	local mean = {}
	local std = {}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = noise[i]:mean()
	   std[i] = noise[{i,{},{}}]:std()
	   noise[{i,{},{}}]:add(-mean[i])
	   noise[{i,{},{}}]:div(std[i])
	end
	--[[
	Gaus=torch.zeros(200,200)
	foyer_x=torch.random(1,200)
	foyer_y=torch.random(1,200)	
	std_x=torch.random(1,5)
	std_y=torch.random(1,5)
	for x=1,200 do
		for y=1,200 do
			Gaus[x][y]=loi_normal(x/20,y/20,foyer_x/20,foyer_y/20,std_x,std_y)
		end
	end
	return im+noise--]]
	return im+noise*coef_DA
end



---------------------------------------------------------------------------------------
-- Function :getBatch(imgs, list, indice, lenght, width, height, Type) 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
-- this function search the indice of associated images and take the corresponding images in imgs which are the loaded images of the folder
function getBatch(imgs, list, indice, lenght, width, height, Type)
	
	if (indice+1)*lenght<#list.im1 then
		start=indice*lenght
	else
		start=#list.im1-lenght
	end
	if Type=="Prop" then
		Batch=torch.Tensor(4, lenght,1, width, height)
	else
		Batch=torch.Tensor(2, lenght,1, width, height)
	end
	
	for i=1, lenght do
		Batch[1][i]=imgs[list.im1[start+i]]
		Batch[2][i]=imgs[list.im2[start+i]]
		if Type=="Prop" then
			Batch[3][i]=imgs[list.im3[start+i]]
			Batch[4][i]=imgs[list.im4[start+i]]
		end
	end

	return Batch

end
---------------------------------------------------------------------------------------
-- Function :getRandomBatchFromSeparateList(imgs1, imgs2, txt1, txt2, lenght, image_width, image_height, Mode, use_simulate_images) 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatchFromSeparateList(Data1,Data2, lenght, Mode)

	local Dim=Data1.images[1]:size()
	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,Dim[1], Dim[2], Dim[3])
	else
		Batch=torch.Tensor(2, lenght,Dim[1], Dim[2], Dim[3])
	end

	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			Set=get_two_Prop_Pair(Data1.Infos, Data2.Infos)
			Batch[1][i]=Data1.images[Set.im1]
			Batch[2][i]=Data1.images[Set.im2]
			Batch[3][i]=Data2.images[Set.im3]
			Batch[4][i]=Data2.images[Set.im4]
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#Data1.images)
			Batch[1][i]=Data1.images[Set.im1]
			Batch[2][i]=Data1.images[Set.im2]
		elseif Mode=="Caus" then
			Set=get_one_random_Caus_Set(Data1.Infos, Data2.Infos)
			Batch[1][i]=Data1.images[Set.im1]
			Batch[2][i]=Data2.images[Set.im2]
		else
			print "getRandomBatchFromSeparateList Wrong mode "
		end
	end
	return Batch

end
---------------------------------------------------------------------------------------
-- Function : getRandomBatch(imgs, txt, lenght, width, height, Mode, use_simulate_images)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatch(Data1, lenght, Mode)
	
	local Dim=Data1.images[1]:size()
	if Mode=="Prop" or Mode=="Rep" then
		Batch=torch.Tensor(4, lenght,Dim[1], Dim[2], Dim[3])
	else
		Batch=torch.Tensor(2, lenght,Dim[1], Dim[2], Dim[3])
	end
	
	for i=1, lenght do
		if Mode=="Prop" or Mode=="Rep" then
			Set=get_one_random_Prop_Set(Data1.Infos)
			Batch[1][i]=Data1.images[Set.im1]
			Batch[2][i]=Data1.images[Set.im2]
			Batch[3][i]=Data1.images[Set.im3]
			Batch[4][i]=Data1.images[Set.im4]
		elseif Mode=="Temp" then
			Set=get_one_random_Temp_Set(#Data1.images)
			Batch[1][i]=Data1.images[Set.im1]
			Batch[2][i]=Data1.images[Set.im2]
		elseif Mode=="Caus" then
			Set=get_one_random_Caus_Set(Data1.Infos,Data1.Infos)
			Batch[1][i]=Data1.images[Set.im1]
			Batch[2][i]=Data1.images[Set.im2]
		else
			print "getRandomBatch Wrong mode "
		end
	end
	return Batch
end


---------------------------------------------------------------------------------------
-- Function :	Have_Todo(list_prior,prior)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Have_Todo(list_prior,prior)
	local answer=false
	if #list_prior~=0 then
		for i=1, #list_prior do
			if list_prior[i]==prior then answer=true end
		end
	end
	return answer
end


---------------------------------------------------------------------------------------
-- Function :	Get_Folder_Name(Log_Folder,Prior_Used)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folder_Name(Log_Folder,list_prior)
	name=''
	if #list_prior~=0 then
		if #list_prior==1 then
			name=list_prior[1].."_Only"
		elseif #list_prior==4 then
			name='Everything'
		else
			name=list_prior[1]
			for i=2, #list_prior do
				name=name..'_'..list_prior[i]
			end
		end
	end
	return Log_Folder..name..'/'
end



---------------------------------------------------------------------------------------
-- Function :copy_weight(model, AE)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function copy_weight(model, AE)
	model:get(1).weight:copy(AE:get(1).weight)
	model:get(4).weight:copy(AE:get(5).weight)
	return model
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function real_loss(txt,use_simulate_images)

	local REP_criterion=get_Rep_criterion()
	local PROP_criterion=get_Prop_criterion()
	local CAUS_criterion=get_Caus_criterion()
	local TEMP_criterion=nn.MSDCriterion()
	
	local truth=getTruth(txt,use_simulate_images)

	local temp_loss=0
	local prop_loss=0
	local rep_loss=0
	local caus_loss=0

	local nb_sample=100

	for i=0, nb_sample do
		Set_prop=get_one_random_Prop_Set(txt ,use_simulate_images)
		Set_temp=get_one_random_Temp_Set(#truth)
		Caus_temp=get_one_random_Caus_Set(txt, txt, use_simulate_images)

		joint1=torch.Tensor(1)
		joint2=torch.Tensor(1)
		joint3=torch.Tensor(1)
		joint4=torch.Tensor(1)

		joint1[1]=truth[Caus_temp.im1]
		joint2[1]=truth[Caus_temp.im2]		
		caus_loss=caus_loss+CAUS_criterion:updateOutput({joint1, joint2})

		joint1[1]=truth[Set_temp.im1]
		joint2[1]=truth[Set_temp.im2]		
		temp_loss=temp_loss+TEMP_criterion:updateOutput({joint1, joint2})

		joint1[1]=truth[Set_prop.im1]
		joint2[1]=truth[Set_prop.im2]
		joint3[1]=truth[Set_prop.im3]
		joint4[1]=truth[Set_prop.im4]
		prop_loss=prop_loss+PROP_criterion:updateOutput({joint1, joint2, joint3, joint4})
		rep_loss=rep_loss+REP_criterion:updateOutput({joint1, joint2, joint3, joint4})
	end

	return temp_loss/nb_sample, prop_loss/nb_sample, rep_loss/nb_sample, caus_loss/nb_sample
end



---------------------------------------------------------------------------------------
-- Function : load_list(list,lenght,height)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_list(list,lenght,height, train)
	local im={}
	local lenght=lenght or 200
	local height=height or 200
	for i=1, #list do
		table.insert(im,getImage(list[i],lenght,height,train))
	end 
	return im
end



---------------------------------------------------------------------------------------
-- Function : load_list(list,lenght,height)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_Part_list(list,txt,txt_reward,im_lenght,im_height,nb_part,part,train,txt_state)
	local im={}
	local Infos={dx={},dy={},dz={},reward={}}
	local im_lenght=im_lenght or 200
	local im_height=im_height or 200
	local list_lenght = torch.floor(#list/nb_part)
	local start=list_lenght*part +1
	local Infos,ThereIsReward=getInfos(txt,txt_reward,start,list_lenght,txt_state)
	for i=start, start+list_lenght do
		table.insert(im,getImage(list[i],im_lenght,im_height,train))
	end 
	return {images=im,Infos=Infos},ThereIsReward
end


-- pb si pas de reward....
function getInfos(txt,txt_reward,start,lenght,txt_state)
	local Infos={dx={},dy={},dz={},reward={}}
	local dx=2
	local dy=3
	local dz=4
	local reward_indice=2

local reward=0--!!!new
local tensor_state, label=tensorFromTxt(txt_state)

	local tensor, label=tensorFromTxt(txt)
	local tensor_reward, label=tensorFromTxt(txt_reward)
	local ThereIsReward=false
	for i=start, start+lenght do
		table.insert(Infos.dx,tensor[i][dx])
		table.insert(Infos.dy,tensor[i][dy])
		table.insert(Infos.dz,tensor[i][dz])

if math.floor(tensor_state[i][dx]*100)%20==0 or math.floor(tensor_state[i][dy]*100)%20==0 or math.floor(tensor_state[i][dz]*100)%20==0 then 
	ThereIsReward=true
	reward=1
else
	reward=0
 end
table.insert(Infos.reward,reward)
--!!!!!!!!!!!!!!table.insert(Infos.reward,tensor_reward[i][reward_indice])
--print(tensor_reward[i][reward_indice])
		

--!!!!!!!!!!!!!!if tensor_reward[i][reward_indice]==1 then ThereIsReward=true end
       end

	return Infos,ThereIsReward

end

---------------------------------------------------------------------------------------
-- Function : getImage(im,length,height,SpacialNormalization)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getImage(im,length,height, coef_DA)
	if im=='' or im==nil then return nil end
	local image1=image.load(im,3,'float')
	local format=length.."x"..height
	local img1_rsz=image.scale(image1,format)
	return preprocessing(img1_rsz,length,height, coef_DA) 
end


