
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

--net = torch.load('model-test.t7'):double()
net = torch.load('./Supervise/Log/22-09/Save22-09.t7'):double()
print('net\n' .. net:__tostring());
--net_color = torch.load('./Log/19-09-WODA/Prop_Rep_Caus/Save19-09-WODA.t7'):double()

--cutorch.setDevice(2) 

local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
local last_indice=#list_folders_images
local list_truth=images_Paths(list_folders_images[last_indice])


txt_test=list_txt_state[last_indice]
txt_reward_test=list_txt_button[last_indice]
nbIm=torch.floor(#list_truth/10)
Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,image_width,image_height,10,0,false)
Data_color=load_Part_list(list_truth,txt_test,txt_reward_test,image_width,image_height,10,0,true)
imgs=Data_test.images
imgs_color=Data_color.images
for i=1, nbIm do
	Batch=torch.Tensor(1,3, 200, 200)
	Batch[1]=imgs[i]
	net:forward(Batch)
	--image.display{image=net:get(19).output[1], nrow=16,  zoom=10, legend="image"..i}
	map=net:get(17).output[1]
	tensor_1=create_superposition(Batch[1],map[1])
	--tensor_2=create_superposition(Batch[1],map[2])
	--tensor_3=create_superposition(Batch[1],map[3])

----------color-------------
--[[
	Batch_color=torch.Tensor(1,3, 200, 200)
	Batch_color[1]=imgs_color[i]
	net_color:forward(Batch_color)
	--image.display{image=net:get(19).output[1], nrow=16,  zoom=10, legend="image"..i}
	map_color=net_color:get(17).output[1]
	
	tensor_color=create_superposition(Batch_color[1],map_color[1])

	out=torch.cat(tensor,tensor_color,2)
--]]

	--out=torch.cat(torch.cat(tensor_1,tensor_2,2),tensor_3,2)
	filename=paths.home.."/Bureau/Resultat_non_supervise/22-09/image"..i..".jpg"
	image.save(filename,tensor_1)
	xlua.progress(i, nbIm)
end

im=net:get(17).output[1][1]
min=im:min()
max=im:max()
im:add(-min)
im:mul(256/(max-min))
im=im:ceil()
	
prob=torch.zeros(256)
for j=1,256 do
	for a=1, im:size(1) do
		for b=1, im:size(2) do
			if a~=1 and im[a][b]-im[a-1][b]==j then
				prob[j]=prob[j]+1
			end
			if b~=1 and im[a][b]-im[a][b-1]==j then
				prob[j]=prob[j]+1
			end
			if a~=im:size(1) and im[a][b]-im[a+1][b]==j then
				prob[j]=prob[j]+1
			end
			if b~=im:size(2) and im[a][b]-im[a][b+1]==j then
				prob[j]=prob[j]+1
			end
		end
	end
end
prob=prob/(im:size(1)*im:size(2))

entropy=0
for j=1, 256 do
	if prob[j] ~=0 then
		entropy=entropy-prob[j]*math.log(prob[j])
	end
end
print("entropy")
print(entropy)

--[[
path_test="/home/lesort/TrainTorch/Kaggle/PreprocessedData/Train/epoch0/c0/img_208.jpg"

image1=getImage(path_test)
net:forward(Data1)
image.display{image=net:get(14).output[1], nrow=8,  zoom=4, legend="image"}
--]]


