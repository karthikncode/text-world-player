-- Framework for tutorial

-- Layer to create quests and act as middle-man between Evennia and Agent
require 'utils'
local _ = require 'underscore'
local DEBUG = false

local DEFAULT_REWARD = -0.01
local JUNK_CMD_REWARD = -0.1
local STEP_COUNT = 0 -- count the number of steps in current episode
local BRIDGE_PASSED = false

--Simple quests
-- quests = {'You are hungry.','You are sleepy.', 'You are bored.', 'You are getting fat.'}

--(somewhat) complex quests
-- quests = {'You are not sleepy but hungry.',
-- 					'You are not hungry but sleepy.',
-- 					'You are not getting fat but bored.',
-- 					'You are not bored but getting fat.'} 

-- quest_actions = {'eat', 'sleep', 'watch' ,'exercise'} -- aligned to quests above
-- quest_checklist = {}
-- mislead_quest_checklist = {}
-- rooms = {'Living', 'Garden', 'Kitchen','Bedroom'}

actions = {'look', 'get', 'light', 'stab', 'climb', 'move', 'go'} -- hard code in
-- action 'go' is treated specially - no need to output the word 'go' in the command
objects = {'here', 'foggy tentacles'} -- read rest from build file
exits = {'east','west','north','south',
				 'up','down',
			   'northern path',
				 'back to cliff',
				 'enter',
				 'leave',
				 'old bridge',
				 'hole into cliff',
				 'climb the chain',
				 'bridge',
				 'standing archway',
				 'along inner wall',
				 'ruined gatehouse',
				 'castle corner',
				 'gatehouse',
				 'courtyard',
				 'temple',
				 'stairs down',
				 'blue bird tomb',
				 'tomb of woman on horse',
				 'tomb of the crowned queen',
				 'tomb of the shield',
				 'tomb of the hero',
				 'antechamber'
				}
objects = TableConcat(objects, exits)

symbols = {}
symbol_mapping = {}
object_mapping = {}

NUM_ROOMS = 4

local current_room_description = ""


function login(user, password)
	local num_rooms = 4
	local pre_login_text = data_in()	
	print(pre_login_text)	
	sleep(1)
	data_out('connect ' .. user .. ' ' .. password)
	data_out('@quell') -- remove superuser permissions, if any
end

--Function to parse the output of the game (to extract rewards, etc. )
-- TODO - customize for tutorial_world
function parse_game_output(text)
	-- extract REWARD if it exists
	-- text is a list of sentences
	-- print("Parsing", text)
	local reward = nil
	local text_to_agent = ""
	local running_text = ""
	local exits = {}
	local objects_available = {}
	for i=1, #text do
		if string.match(text[i], '<EOM>') then
			text_to_agent = running_text
			running_text = ""

		--these are specific lines for reward, objects, exits, etc.	
		elseif string.match(text[i], "REWARD") then
			if not BRIDGE_PASSED or not string.match(text[i], "REWARD_bridge") then
				if reward then
					reward = reward + tonumber(string.match(text[i], "[-%d]+"))
				else
					reward = tonumber(string.match(text[i], "[-%d]+"))
				end				
			end

			--prevent from getting the same reward again for passing the bridge
			if string.match(text[i], "REWARD_bridge") then
				BRIDGE_PASSED = true
			end
		elseif string.match(text[i], "Exits:") then
			exits = text[i]:gsub('Exits:', ''):lower():split(',')

		elseif string.match(text[i], 'You see:') then
			objects_available = text[i]:gsub('You see:', ''):lower():split(',')

		-- Incorrect command cases
		elseif 	 string.match(text[i], 'not available') 
					or string.match(text[i], 'not find')
					or string.match(text[i], "You can't get that.") then
			if reward then
				reward = reward + JUNK_CMD_REWARD			
			else
				reward = JUNK_CMD_REWARD			
			end
		-- normal line of text description			
		else
			running_text = running_text .. ' ' .. text[i]
		end
	end
	if not reward then
		reward = DEFAULT_REWARD
	end

	-- if reward == JUNK_CMD_REWARD then
	-- 	text_to_agent = current_room_description
	-- else
	-- 	current_room_description = text_to_agent -- cache
	-- end

	return text_to_agent, reward, exits, objects_available
end


--take a step in the game
function step_game(action_index, object_index, gameLogger)
	local command = build_command(actions[action_index], objects[object_index], gameLogger)
	data_out(command)
	if DEBUG then 
		print(actions[action_index] .. ' ' .. objects[object_index])
	end
	STEP_COUNT = STEP_COUNT + 1
	return getState(gameLogger)
end

-- starts a new game
function newGame(gameLogger)
	data_out("@teleport tut#02") --teleport to Cliff by the coast	
	data_out("look") 
	sleep(0.5)
	STEP_COUNT = 0
	BRIDGE_PASSED = false
	return getState(gameLogger)
end

-- build game command to send to the game
-- TODO:  need to handle special cases for 'go', etc.
function build_command(action, object, logger)
	if logger then
		logger:write(">>" .. action .. ' '.. object..'\n')
	end

	if action == 'go' then
		return object
	end
	if not object or not action then
		print(action, object)
	end
	return action .. ' ' ..object
end

-- TODO
function parseLine(line)
	-- parse line to update symbols and symbol_mapping
	-- IMP: make sure we're using simple english - ignores punctuation, etc.
	local sindx
	for word in line:gmatch('%a+') do
		word = word:lower()	
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			symbol_mapping[word] = sindx
		end
	end
end

-- TODO
-- read in text data from file with sentences (one sentence per line) - nicely tokenized
function makeSymbolMapping(filename)
	local file = io.open(filename, "r");
	local data = {}
	local parts
	for line in file:lines() do
		line = string.gsub(line, "{.", "")
		if line:starts('@create/drop') then
			local object = _.join(_.rest(split(line, '%S+')), ' '):match('[%a ]+'):lower():trim() -- extract object name
			table.insert(objects, object)
		elseif not line:starts('#') then
			parseLine(line)
		end
	end

	--create a map for objects
	for i, object in pairs(objects) do
		object_mapping[object] = i
	end
	
	--add aliases for exits (not best way but in order to speed up gameplay)
	object_mapping['ruined temple'] = object_mapping['temple']
	object_mapping['overgrown courtyard'] = object_mapping['courtyard']
	object_mapping['bridge over the abyss'] = object_mapping['bridge']
	object_mapping['up the stairs to ruined temple'] = object_mapping['up']
end

-- input_text is just a string with the room description
function convert_text_to_bow(input_text)
	local vector = torch.zeros(#symbols)
	line = input_text
	local list_words = split(line, "%a+")
	for i=1,#list_words do			
		local word = list_words[i]
		word = word:lower()
		--ignore words not in vocab
		if symbol_mapping[word] then	
			vector[symbol_mapping[word]] = vector[symbol_mapping[word]] + 1
		else
			-- print("<"..word .. '> not in vocab')
		end

	end

	return vector
end


-- for recurrent and other networks
-- assumes that the symbol mapping has already been created
-- STATE_DIM = max desc/quest length
function convert_text_to_ordered_list(input_text)
	local NULL_INDEX = #symbols + 1
	local vector = torch.ones(STATE_DIM) * NULL_INDEX
	local REVERSE = true --reverse the order of words to have padding in beginning
	cnt=1
	
	line = input_text
	local list_words = split(line, "%a+")
	for i=1,#list_words do			
		local word = list_words[i]
		word = word:lower()
		if REVERSE then cnt2 = STATE_DIM+1-cnt else cnt2 = cnt end
		--ignore words not in vocab
		if symbol_mapping[word] then
			vector[cnt2] = symbol_mapping[word]
		else
			print(word .. ' not in vocab')
		end
		cnt=cnt+1
	end

	-- return reverse_tensor(vector)
	return vector
end


-------------------------VECTOR function -------------------------
if RECURRENT == 1 then
	vector_function = convert_text_to_ordered_list
else
	vector_function = convert_text_to_bow
end
-------------------------------------------------------------------

--function to find the indices of objects given their textual forms
function findObjectIndices(list)
	local indices = {}
	for i,s in pairs(list) do
		s = s:trim()
		local index = object_mapping[s]
		if index then
			table.insert(indices, index)
		else
			print("Object not found:<".. s .. '>')
		end
	end
	if #indices>0 then
		return indices
	else
		return nil
	end
end


function getState(logger, print_on)
	local terminal = (STEP_COUNT >= MAX_STEPS)
	local inData = data_in()
	while #inData == 0 or not string.match(inData[#inData],'<EOM>') do	
		TableConcat(inData, data_in())
	end

	-- print('indata', inData)
	-- print('indata2', inData2)
	local text, reward, exits, objects_available = parse_game_output(inData)		

	-- look only if command was junk
	if reward == JUNK_CMD_REWARD then		
		data_out('look')
		local inData2 = data_in()
		while #inData2 == 0 or not string.match(inData2[#inData2],'<EOM>') do	
			TableConcat(inData2, data_in())
		end
		local text2, tmp_reward, exits2, objects_available2 = parse_game_output(inData2) -- the room description after 'looking'
		text = text .. text2
		exits = exits2
		objects_available = objects_available2
		reward = reward + tmp_reward
	end

	if DEBUG or print_on then
		print(text, reward)
		print("Exits: ", exits)
		print("Objects: ", objects_available)
		sleep(0.1)
		if reward > 0 or reward <= -1 then
			print(text, reward)
			sleep(0.1)
		end
	end
	if reward >= 10 then
		terminal = true
		-- sleep(5)
	end
	if reward >=1  then
		print(text, reward)
	end

	local vector = vector_function(text)
	-- print("Exits: ", exits)
	-- print("Objects: ", objects_available)
	local available_objects
	if #exits + #objects_available > 0 then
		available_objects = findObjectIndices(TableConcat(exits, objects_available))
		-- print("available objects" , available_objects)
	end

	if logger then
		if type(text) == 'table' then
			logger:write(table.concat(text, ' '), '\n')
		else
			logger:write(text, '\n')
		end
		logger:write('Reward: '..reward, '\n')
		if terminal then
			logger:write('****************************\n\n')
		end
	end	
	return vector, reward, terminal, available_objects
end


function getActions()
	return actions
end

function getObjects()
	return objects
end


return {
	makeSymbolMapping = makeSymbolMapping,
	getActions = getActions, 
	getObjects = getObjects, 
	getState = getState, 
	step = step_game,
	newGame = newGame,
	nextRandomGame = nextRandomGame,
}