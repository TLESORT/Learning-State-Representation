

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

local use_simulate_images=true
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files()
local last_indice=#list_folders_images
local list=images_Paths(list_folders_images[last_indice])


indice_test=4 --nbList
list_truth=images_Paths(list_folders_images[indice_test])
txt_test=list_txt_state[indice_test]
txt_reward_test=list_txt_button[indice_test]
Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,200,200,50,0,true,txt_test)
im1=clampImage(Data_test.images[40])
Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,200,200,50,0,true,txt_test)
im2=clampImage(Data_test.images[40])
Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,200,200,50,0,true,txt_test)
im3=clampImage(Data_test.images[40])
Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,200,200,50,0,true,txt_test)
im4=clampImage(Data_test.images[40])
Data_test=load_Part_list(list_truth,txt_test,txt_reward_test,200,200,50,0,false,txt_test)
im5=clampImage(Data_test.images[40])

noise=torch.rand(3,200,200)
noise=noise-noise:mean()+im5:mean()
noise=(noise/noise:std())*im5:std()
im6=clampImage(noise+im5)


haut=torch.cat(im5,torch.cat(im1,im2,3),3)
bas=torch.cat(im6,torch.cat(im3,im4,3),3)
im=torch.cat(haut,bas,2)
image.display{im}

filename=paths.home.."/Bureau/imageDAtaAugmentation.jpg"
image.save(filename,im)
