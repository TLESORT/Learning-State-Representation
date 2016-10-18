
---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input (Path): path of a Folder which contained jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
function images_Paths(Path)
	local listImage={}
	for file in paths.files(Path) do
	   -- We only load files that match the extension
	   if file:find('jpg' .. '$') then
	      -- and insert the ones we care about in our table
	      table.insert(listImage, paths.concat(Path,file))
	   end
	   
	end
	table.sort(listImage)
	return listImage
end


---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function txt_path(Path,including)
	local including=including or ""
	local txt=nil
	for file in paths.files(Path) do
	   if file:find(including..'.txt' .. '$') then
	      txt=paths.concat(Path,file)
	   end
	end
	return txt
end

---------------------------------------------------------------------------------------
-- Function : 
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folders(Path, including, excluding,list)
	local list=list or {}
	local incl=including or ""
	local excl=excluding or "uyfouhjbhytfoughl" -- random motif
	for file in paths.files(Path) do
	   -- We only load files that match 2016 because we know that there are the folder we are interested in
	   if file:find(incl) and (not file:find(excl)) then
	      -- and insert the ones we care about in our table
	      table.insert(list, paths.concat(Path,file))
	   end
	end
	return list
end


---------------------------------------------------------------------------------------
-- Function : Get_HeadCamera_HeadMvt(use_simulate_images)
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images 
-- Output (list_head_left): list of the images directories path
-- Output (list_txt):  txt list associated to each directories (this txt file contains the grundtruth of the robot position)
---------------------------------------------------------------------------------------
function Get_HeadCamera_View_Files()
	local use_simulate_images=use_simulate_images or false
	local Path
	Path=paths.home.."/data_baxter/"
	Path=paths.home.."/data_baxter/"

	local Paths=Get_Folders(Path,'record')
	list_folder={}
	list_txt_button={}
	list_txt_action={}
	list_txt_state={}

	for i=1, #Paths do
		list_folder=Get_Folders(Paths[i],'recorded','txt',list_folder)
	end

	for i=1, #Paths do
		table.insert(list_txt_button, txt_path(Paths[i],"is_pressed"))
		table.insert(list_txt_action, txt_path(Paths[i],"endpoint_action"))
		table.insert(list_txt_state, txt_path(Paths[i],"endpoint_state"))
	end
	table.sort(list_txt_button)
	table.sort(list_txt_action)
	table.sort(list_txt_state)
	table.sort(list_folder)
	return list_folder, list_txt_action,list_txt_button, list_txt_state
end


---------------------------------------------------------------------------------------
-- Function : tensorFromTxt(path)
-- Input (path) : path of a txt file which contain position of the robot
-- Output (torch.Tensor(data)): tensor with all the joint values (col: joint, lign : indice)
-- Output (labels):  name of the joint
---------------------------------------------------------------------------------------
function tensorFromTxt(path)
    local data, raw = {}, {}
    local rawCounter, columnCounter = 0, 0
    local nbFields, labels, _line = nil, nil, nil

    for line in io.lines(path)  do 
        local comment = false
        if line:sub(1,1)=='#' then 
            comment = true            
            line = line:sub(2)
        end 
        rawCounter = rawCounter +1      
        columnCounter=0
        raw = {}
        for value in line:gmatch'%S+' do
            columnCounter = columnCounter+1
            raw[columnCounter] = tonumber(value)
        end

        -- we check that every row contains the same number of data
        if rawCounter==1 then
            nbFields = columnCounter
        elseif columnCounter ~= nbFields then
            error("data dimension for " .. rawCounter .. "the sample is not consistent with previous samples'")
        end
    
        if comment then labels = raw else table.insert(data,raw) end 
    end
    return torch.Tensor(data), labels
end


---------------------------------------------------------------------------------------
-- Function : get_one_random_Temp_Set(list_im)
-- Input (list_lenght) : lenght of the list of images
-- Output : 2 indices of images which are neightboor in the list (and in time) 
---------------------------------------------------------------------------------------
function get_one_random_Temp_Set(list_lenght)
	indice=torch.random(1,list_lenght-1)
	return {im1=indice,im2=indice+1}
end


function get_one_random_Prop_Set(Infos1)
	local WatchDog=0
	local size1=#Infos1.dx
	local vector=torch.randperm(size1-1)

	while WatchDog<100 do
		local indice1=torch.random(1,size1-1)
		local indice2=indice1+1

		for i=1, size1-1 do
			local id=vector[i]
			local id2=id+1

			if id~=indice1 and arrondit(Infos1.dx[indice1]-Infos1.dx[id])==0 and
				arrondit(Infos1.dy[indice1]-Infos1.dy[id])==0 and
				arrondit(Infos1.dz[indice1]-Infos1.dz[id])==0 then
				return {im1=indice1,im2=indice2,im3=id,im4=id2}
			elseif id2~=indice1 and arrondit(Infos1.dx[indice1]+Infos1.dx[id])==0 and
				arrondit(Infos1.dy[indice1]+Infos1.dy[id])==0 and
				arrondit(Infos1.dz[indice1]+Infos1.dz[id])==0 then
				return {im1=indice1,im2=indice2,im3=id2,im4=id}
			end
		end
		WatchDog=WatchDog+1
	end
	print("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end
---------------------------------------------------------------------------------------
-- Function : get_two_Prop_Pair(txt1, txt2,use_simulate_images)
-- Input (txt1) : path of the file of the first list of joint
-- Input (txt2) : path of the file of the second list of joint
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images (we need this information because the data is not formated exactly the same in the txt file depending on the origin of images)
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Traininng with prop prior. The variation of joint for on pair should be the same as the variation for the second
---------------------------------------------------------------------------------------
function get_two_Prop_Pair(Infos1, Infos2)

	local WatchDog=0
	local ecart=1

	local size1=#Infos1.dx
	local size2=#Infos2.dx

	local vector=torch.randperm(size2-1)

	while WatchDog<100 do
		local indice1=torch.random(1,size1-1)
		local indice2=indice1+1

		for i=1, size2-1 do
			id=vector[i]
			id2=id+1

			if arrondit(Infos1.dx[indice1]-Infos2.dx[id])==0 and
				arrondit(Infos1.dy[indice1]-Infos2.dy[id])==0 and
				arrondit(Infos1.dz[indice1]-Infos2.dz[id])==0 then
				return {im1=indice1,im2=indice2,im3=id,im4=id2}
			elseif  arrondit(Infos1.dx[indice1]+Infos2.dx[id])==0 and
				arrondit(Infos1.dy[indice1]+Infos2.dy[id])==0 and
				arrondit(Infos1.dz[indice1]+Infos2.dz[id])==0 then
				return {im1=indice1,im2=indice2,im3=id2,im4=id}
			end
		end
		WatchDog=WatchDog+1
	end
	print("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

-- I need to search images representing a starting state.
-- then the same action applied to this to state (the same variation of joint) should lead to a different reward.
-- for instance we choose for reward the fact to have a joint = 0

-- NB : the two states will be took in different list but the two list can be the same

local function causality_applicable(Infos1,Infos2,indice1,indice2,id, delta)
	id2=id+delta
	if delta==1 and arrondit(Infos1.dx[indice1]-Infos2.dx[id])==0 and
		arrondit(Infos1.dy[indice1]-Infos2.dy[id])==0 and
		arrondit(Infos1.dz[indice1]-Infos2.dz[id])==0 and
		Infos2.reward[id2]==1 and
		Infos2.reward[id]==0 then
		return true
	elseif delta==2 and arrondit(Infos1.dx[indice1]-Infos2.dx[id]+Infos1.dx[indice1+1]-Infos2.dx[id+1])==0 and
		arrondit(Infos1.dy[indice1]-Infos2.dy[id]+Infos1.dy[indice1+1]-Infos2.dy[id+1])==0 and
		arrondit(Infos1.dz[indice1]-Infos2.dz[id]+Infos1.dz[indice1+1]-Infos2.dz[id+1])==0 and
		Infos2.reward[id2]==1 and
		Infos2.reward[id]==0 then
		return true
	else
		return false
	end
end
local function causality_applicable2(Infos1,Infos2,indice1,indice2,id, delta)
	id2=id+delta
	if delta==1 and arrondit(Infos1.dx[indice1]+Infos2.dx[id])==0 and
		arrondit(Infos1.dy[indice1]+Infos2.dy[id])==0 and
		arrondit(Infos1.dz[indice1]+Infos2.dz[id])==0 and
		Infos2.reward[id]==1 and
		Infos2.reward[id2]==0 then
		return true
	elseif delta==2 and arrondit(Infos1.dx[indice1]+Infos2.dx[id]+Infos1.dx[indice1+1]+Infos2.dx[id+1])==0 and
		arrondit(Infos1.dy[indice1]+Infos2.dy[id]+Infos1.dy[indice1+1]+Infos2.dy[id+1])==0 and
		arrondit(Infos1.dz[indice1]+Infos2.dz[id]+Infos1.dz[indice1+1]+Infos2.dz[id+1])==0 and
		Infos2.reward[id]==1 and
		Infos2.reward[id2]==0 then
		return true
	else
		return false
	end
end

function get_one_random_Caus_Set(Infos1,Infos2)
	local WatchDog=0
	local dx=2
	local dy=3
	local dz=4

	local size1=#Infos1.dx
	local size2=#Infos2.dx
	vector=torch.randperm(size2-1)

	while WatchDog<1000 do
		repeat
			indice1=torch.random(1,size1-1)
			indice2=indice1+1
		until(Infos1.reward[indice1]==0 and Infos1.reward[indice2]==0)

		for i=1, size2-1 do
			id=vector[i]
			id2=id+1
			if causality_applicable(Infos1,Infos2,indice1,indice2,id, 1) then
				return {im1=indice1,im2=id}
			elseif causality_applicable(Infos1,Infos2,indice1,indice2,id, 2) then
				return {im1=indice1,im2=id}
			elseif causality_applicable2(Infos1,Infos2,indice1,indice2,id, 1) then
				return {im1=indice1,im2=id+1}
			elseif causality_applicable2(Infos1,Infos2,indice1,indice2,id, 2) then
				return {im1=indice1,im2=id+2}
			end
		end
		WatchDog=WatchDog+1
	end
	print("CAUS WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end


---------------------------------------------------------------------------------------
-- Function : getTruth(txt,use_simulate_images)
-- Input (txt) : 
-- Input (use_simulate_images) : 
-- Input (arrondit) :
-- Output (truth): 
---------------------------------------------------------------------------------------
function getTruth(txt_joint, nb_part, part)
	local x=2
	local y=3
	local z=4
	
	local tensor, label=tensorFromTxt(txt_joint)
	local list_lenght = torch.floor((#tensor[{}])[1]/nb_part)
	local start=list_lenght*part +1

	local list_truth={}
	for i=start, start+list_lenght do--(#tensor[{}])[1] do	
		local truth=torch.Tensor(3)
		truth[1]=tensor[i][x]
		truth[2]=tensor[i][y]
		truth[3]=tensor[i][z]
		table.insert(list_truth,truth)
	end
	return list_truth
end


---------------------------------------------------------------------------------------
-- Function : arrondit(value)
-- Input (tensor) : 
-- Input (head_pan_indice) : 
-- Output (tensor): 
---------------------------------------------------------------------------------------
function arrondit(value) --0.05 precision
	floor=math.floor(value*20)/20
	ceil=math.ceil(value*20)/20
	if math.abs(value-ceil)>math.abs(value-floor) then result=floor
	else result=ceil end
	return result
end
