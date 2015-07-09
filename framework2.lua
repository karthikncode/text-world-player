-- Framework for fantasy world

-- Layer to create quests and act as middle-man between Evennia and Agent
require 'utils'
local _ = require 'underscore'
local DEBUG = false

local DEFAULT_REWARD = -0.01
local JUNK_CMD_REWARD = -1
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

actions = {'get', 'light', 'stab', 'climb', 'move', 'go'} -- hard code in
-- action 'go' is treated specially - no need to output the word 'go' in the command
objects = {'here', 'foggy tentacles'} -- read rest from build file
exits = {'east','west','north','south',
				 'up','down',
			   'northern path',
				 'back to cliff',
				 'enter',
				 'leave', --10
				 'hole into cliff',
				 'climb the chain',
				 'bridge',
				 'standing archway',
				 'along inner wall',
				 'ruined gatehouse',
				 'castle corner',
				 'gatehouse',
				 'courtyard',
				 'temple', --20
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


extra_messages = {"You are standing {wvery close to the the bridge's western foundation{n. If you go west you will be back on solid ground ...",
                       "The bridge slopes precariously where it extends eastwards towards the lowest point - the center point of the hang bridge.",
                       "You are {whalfways{n out on the unstable bridge.",
                       "The bridge slopes precariously where it extends westwards towards the lowest point - the center point of the hang bridge.",
                       "You are standing {wvery close to the bridge's eastern foundation{n. If you go east you will be back on solid ground ...",
								"The bridge sways in the wind.", "The hanging bridge creaks dangerously.",
                "You clasp the ropes firmly as the bridge sways and creaks under you.",
                "From the castle you hear a distant howling sound, like that of a large dog or other beast.",
                "The bridge creaks under your feet. Those planks does not seem very sturdy.",
                "Far below you the ocean roars and throws its waves against the cliff, as if trying its best to reach you.",
                "Parts of the bridge come loose behind you, falling into the chasm far below!",
                "A gust of wind causes the bridge to sway precariously.",
                "Under your feet a plank comes loose, tumbling down. For a moment you dangle over the abyss ...",
                "The section of rope you hold onto crumble in your hands, parts of it breaking apart. You sway trying to regain balance.",
							 "Suddenly the plank you stand on gives way under your feet! You fall!",
               "\nYou try to grab hold of an adjoining plank, but all you manage to do is to ",
               "divert your fall westwards, towards the cliff face. This is going to hurt ... ",
               "\n ... The world goes dark ...\n\n",
               "It is pitch black. You are likely to be eaten by a grue.",
                 "It's pitch black. You fumble around but cannot find anything.",
                 "You don't see a thing. You feel around, managing to bump your fingers hard against something. Ouch!",
                 "You don't see a thing! Blindly grasping the air around you, you find nothing.",
                 "It's totally dark here. You almost stumble over some un-evenness in the ground.",
                 "You are completely blind. For a moment you think you hear someone breathing nearby ... \n ... surely you must be mistaken.",
                 "Blind, you think you find some sort of object on the ground, but it turns out to be just a stone.",
                 "Blind, you bump into a wall. The wall seems to be covered with some sort of vegetation, but its too damp to burn.",
                 "You can't see anything, but the air is damp. It feels like you are far underground.",
								"You don't want to stumble around in blindness anymore. You already ",
                      "found what you need. Let's get light already!",
								"Your fingers bump against a splinter of wood in a corner. It smells of resin and seems dry enough to burn! ",
                    "You pick it up, holding it firmly. Now you just need to {wlight{n it using the flint and steel you carry with you."}


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
					reward = reward + tonumber(string.match(text[i], "[-%.%d]+"))
				else
					reward = tonumber(string.match(text[i], "[-%.%d]+"))
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
					or string.match(text[i], "You can't get that.") 
					or string.match(text[i], "you cannot") 
					or string.match(text[i], "splinter is already burning")
					or string.match(text[i], "is no way")
					or string.match(text[i], "not uproot them") then
			if reward then
				reward = reward + JUNK_CMD_REWARD			
			else
				reward = JUNK_CMD_REWARD			
			end
			running_text = running_text .. ' ' .. text[i]
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
	return getState(gameLogger, actions[action_index])
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

	--extra messages
	for i, line in pairs(extra_messages) do
		line = string.gsub(line, "{.", "")
		parseLine(line)
	end

	--create a map for objects
	for i, object in pairs(objects) do
		object_mapping[object] = i
	end
	
	--add aliases for exits (not best way but in order to speed up gameplay)
	object_mapping['old bridge'] = object_mapping['bridge']
	object_mapping['ruined temple'] = object_mapping['temple']
	object_mapping['overgrown courtyard'] = object_mapping['courtyard']
	object_mapping['bridge over the abyss'] = object_mapping['bridge']
	object_mapping['up the stairs to ruined temple'] = object_mapping['up']
	object_mapping['back to antechamber'] = object_mapping['antechamber']
end

-- input_text is just a string with the room description
function convert_text_to_bow(input_text)
	local vector = torch.zeros(#symbols)
	local line = string.gsub(input_text, "{.","")
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

-- Args: {
--	1: desc of room
--	2: quest desc
-- }
function convert_text_to_bigram(input_text)
	local vector = torch.zeros(#symbols*#symbols)
	for j, line in pairs(input_text) do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words-1 do			
			local word = list_words[i]
			word = word:lower()
			next_word = list_words[i+1]:lower()
			--ignore words not in vocab
			if symbol_mapping[word] and symbol_mapping[next_word] then	
				vector[(symbol_mapping[word]-1)* (#symbols) + symbol_mapping[next_word]] 
					= vector[(symbol_mapping[word]-1)* (#symbols) + symbol_mapping[next_word]]  + 1
			else
				print(word .. ' not in vocab')
			end

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
	
	local line = string.gsub(input_text, "{.","")
	local list_words = split(line, "%a+")
	for i=1,math.min(#list_words, STATE_DIM) do			
		local word = list_words[i]
		word = word:lower()
		if REVERSE then cnt2 = STATE_DIM+1-cnt else cnt2 = cnt end
		--ignore words not in vocab
		if symbol_mapping[word] then
			vector[cnt2] = symbol_mapping[word]
		else
			-- print(word .. ' not in vocab')
			-- print(input_text)
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
	-- vector_function = convert_text_to_bow
	vector_function = convert_text_to_bigram
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


function getState(logger, prev_command)
	local terminal = (STEP_COUNT >= MAX_STEPS)
	local inData = data_in()
	while #inData == 0 or not string.match(inData[#inData],'<EOM>') do	
		TableConcat(inData, data_in())
	end

	-- print('indata', inData)
	-- print('indata2', inData2)
	local text, reward, exits, objects_available = parse_game_output(inData)		

	-- look only if command was junk
	if reward == JUNK_CMD_REWARD or (prev_command and prev_command ~='go') then		
		data_out('look')
		local inData2 = data_in()
		while #inData2 == 0 or not string.match(inData2[#inData2],'<EOM>') do	
			TableConcat(inData2, data_in())
		end
		local text2, tmp_reward, exits2, objects_available2 = parse_game_output(inData2) -- the room description after 'looking'
		if text ~= text2 then		
			text = text .. text2
		end
		exits = exits2
		objects_available = objects_available2
		reward = reward + tmp_reward	
	end

	if DEBUG then
		print(text, reward)
		print("Exits: ", exits)
		print("Objects: ", objects_available)
		-- sleep(0.1)
		if reward > 0 or reward <= -1 then
			print("DEBUG", text, reward)
			-- sleep(5)
		end
	end
	if reward > 9 then
		terminal = true
		-- sleep(5)
	end
	if reward > 0.9 then
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
