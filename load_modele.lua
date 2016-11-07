
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
require "Get_Images_Set"
require 'priors'

function rescale_10_200(im)
	img_rescale=torch.Tensor(3,200, 200)
	for i=1, 3 do
		for j=1,200 do
			for l=1,200 do
				img_rescale[i][j][l]=im[math.ceil(j/20)][math.ceil(l/20)]
			end
		end
	end

	return img_rescale
end

function rescale_47_188(im)
	img_rescale=torch.Tensor(3,188, 188)
	for i=1, 3 do
		for j=1,188 do
			for l=1,188 do
				img_rescale[i][j][l]=im[math.ceil(j/4)][math.ceil(l/4)]
			end
		end
	end

	return img_rescale
end

local function clampImage(tensor)
   if tensor:type() == 'torch.ByteTensor' then
      return tensor
   end
    
   local a = torch.Tensor():resize(tensor:size()):copy(tensor)
   min=a:min()
   max=a:max()
   a:add(-min)
   a:mul(1/(max-min))         -- remap to [0-1]
   return a
end

local function create_superposition(im,map)
	map_rsz=clampImage(rescale_47_188(map))
	im=clampImage(image.scale(im,188,188))
	salience=clampImage(torch.cmul(map_rsz,im))
	return torch.cat(torch.cat(map_rsz,salience,3),im,3)
end

local function topActivation(net,imgs,level)
	for i=1, nbIm do
		Batch=torch.Tensor(2,3, 200, 200)
		Batch[1]=imgs[i]
		Batch[2]=imgs[i]
		net:forward(Batch)
		--image.display{image=net:get(19).output[1], nrow=16,  zoom=10, legend="image"..i}
		map_1=net:get(16).output[1][1]
		map_2=net:get(17).output[1][1]
		map_3=net:get(18).output[1][1]
		tensor_1=create_superposition(Batch[1],map_1)
		tensor_2=create_superposition(Batch[1],map_2)
		tensor_3=create_superposition(Batch[1],map_3)
		local out=torch.cat(torch.cat(tensor_1,tensor_2,2),tensor_3,2)
		filename=paths.home.."/Bureau/Resultat_supervise/18-10/image"..i..".jpg"
		image.save(filename,out)
		xlua.progress(i, nbIm)
	end
end

local function getBestActivation(net,imgs,level,zoom)
	local zoom=zoom or 1
	Batch=torch.Tensor(1,3, 200, 200)
	Batch[1]=imgs[1]
	net:forward(Batch)
	maps=net:get(level).output[1]
	mean=maps:mean(1):sort()
	local best=torch.zeros(3)
	local indice=maps:size(1)
	local eps=0.0001
	local res=image.y2jet((maps[indice])*10/maps[indice]:max()+eps)
	if indice>9 then
		last=8
	else
		last=2
	end
	for i=1,last do
		res=torch.cat(res,image.y2jet((maps[indice-i])*10/maps[indice-i]:max()+eps),3)

	end
	image.display{image=res,zoom=zoom}
end

local function showWeight(net,level,zoom)
	local zoom=zoom or 10
	W=net:get(1).weight
	D1=W:size(1)
	D2=W:size(2)
	im=torch.cat(W[1][1],torch.ones(3),2)
	for i=2,D1 do
		im=torch.cat(torch.cat(im,W[i][1],2),torch.ones(3),2)
	end
	im2=torch.cat(W[1][2],torch.ones(3),2)
	for i=2,D1 do
		im2=torch.cat(torch.cat(im2,W[i][2],2),torch.ones(3),2)
	end
	im3=torch.cat(W[1][3],torch.ones(3),2)
	for i=2,D1 do
		im3=torch.cat(torch.cat(im3,W[i][3],2),torch.ones(3),2)
	end
	im=torch.cat(im,torch.ones(im:size(2)),1)
	im2=torch.cat(im2,torch.ones(im2:size(2)),1)
	res=torch.cat(torch.cat(im,im2,1),im3,1)
	image.display{image=res,zoom=zoom}
end
name="/home/timothee/git/Learning_State_Representation/Supervise/Log/18-10/Save18-10.t7"
local net = torch.load(name):double()
print('net\n' .. net:__tostring());
--net_color = torch.load('./Log/19-09-WODA/Prop_Rep_Caus/Save19-09-WODA.t7'):double()


local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
local last_indice=#list_folders_images
indice=4
local list_truth=images_Paths(list_folders_images[indice])

txt_test=list_txt_state[indice]
txt_reward_test=list_txt_button[indice]
nbIm=torch.floor(#list_truth/50)
txt_test=list_txt_state[indice]

Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,image_width,image_height,50,1,true,txt_test)
--Data_color=load_Part_list(list_truth,txt_test,txt_reward_test,image_width,image_height,10,0,true)
--getBestActivation(net,Data_test.images,3,1)
--getBestActivation(net,Data_test.images,6,1)
--getBestActivation(net,Data_test.images,9,1)
--getBestActivation(net,Data_test.images,13,2)
--getBestActivation(net,Data_test.images,17,4)
--showWeight(net,1,10)

topActivation(net,Data_test.images,17)
----------color-------------
--out_color=topActivation(net_color,Data_color.images,17)

