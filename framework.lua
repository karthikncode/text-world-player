-- Layer to create quests and act as middle-man between Evennia and Agent
require 'utils'
local underscore = require 'underscore'
local DEBUG = false

local DEFAULT_REWARD = -0.05
local STEP_COUNT = 0 -- count the number of steps in current episode
local MAX_STEPS = 500

quests = {'You are hungry.','You are sleepy.', 'You are bored.', 'You are getting fat.'}
quest_actions = {'eat', 'sleep', 'watch' ,'exercise'} -- aligned to quests above
quest_checklist = {}
quest_levels = 2 --number of levels in any given quest
rooms = {'Living', 'Garden', 'Kitchen','Bedroom'}

actions = {"eat", "sleep", "watch", "exercise", "go"} -- hard code in
objects = {'north','south','east','west'} -- read rest from build file

symbols = {}
symbol_mapping = {}

NUM_ROOMS = 4

local current_room_description = ""

function random_teleport()
	local room_index = torch.random(1, NUM_ROOMS)
	data_out('@tel tut#0'..room_index)
	sleep(0.1)
	data_in()
	data_out('l')
	if DEBUG then
		print('Start Room : ' .. room_index ..' ' .. rooms[room_index])
	end
end

function random_quest()
	indxs = torch.randperm(#quests)
	for i=1,quest_levels do
		local quest_index = indxs[i]
		-- local quest_index = torch.random(1, #quests)
		quest_checklist[#quest_checklist+1] = quest_index
	end
	if DEBUG then
		print("Start quest", quests[quest_checklist[1]], quest_actions[quest_checklist[1]])
	end
end

function login(user, password)
	local num_rooms = 4
	local pre_login_text = data_in()	
	print(pre_login_text)	
	sleep(1)
	data_out('connect ' .. user .. ' ' .. password)
end

--Function to parse the output of the game (to extract rewards, etc. )

function parse_game_output(text)
	-- extract REWARD if it exists
	-- text is a list of sentences
	local reward = nil
	local text_to_agent = {current_room_description, quests[quest_checklist[1]]}
	for i=1, #text do
		if i < #text  and string.match(text[i], '<EOM>') then
			text_to_agent = {current_room_description, quests[quest_checklist[1]]}
		elseif string.match(text[i], "REWARD") then
			if string.match(text[i], quest_actions[quest_checklist[1]]) then
				reward = tonumber(string.match(text[i], "%d+"))
			end
		else
			--IMP: only description and quest are necessary (for now)
			--table.insert(text_to_agent, text[i])
		end
	end
	if not reward then
		reward = DEFAULT_REWARD
	end
	return text_to_agent, reward	
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


-- TODO
function nextRandomGame()
end

-- TODO
function newGame(gameLogger)

	quest_checklist = {}
	STEP_COUNT = 0
	random_teleport()
	random_quest()

	if gameLogger then
	end

	return getState(gameLogger)
end

-- build game command to send to the game
function build_command(action, object, logger)
	if logger then
		logger:write(">>" .. action .. ' '.. object..'\n')
	end
	return action .. ' ' ..object
end


function parseLine( list_words, start_index)
	-- parse line to update symbols and symbol_mapping
	-- IMP: make sure we're using simple english - ignores punctuation, etc.
	local sindx	
	start_index = start_index or 1
	for i=start_index,#list_words do			
		word = split(list_words[i], "%a+")[1]
		word = word:lower()	
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			symbol_mapping[word] = sindx
		end
	end
end

function addQuestWordsToVocab()
	for i, quest in pairs(quests) do
		parseLine(split(quest, "%a+"), 1)
	end
end


-- read in text data from file with sentences (one sentence per line) - nicely tokenized
function makeSymbolMapping(filename)
	local file = io.open(filename, "r");
	local data = {}
	local parts
	for line in file:lines() do
		list_words = split(line, "%S+")
		if list_words[1] == '@detail' or list_words[1] == '@desc' then
			parseLine(list_words, 4)
		elseif list_words[1] == '@create/drop' then
			-- add to actionable objects			
			table.insert(objects, split(list_words[2], "%a+")[1])
		end
	end
	addQuestWordsToVocab()
end

-- Args: {
--	1: desc of room
--	2: quest desc
-- }
function convert_text_to_bow(input_text)
	local vector = torch.zeros(#symbols)
	for j, line in pairs(input_text) do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do			
			local word = list_words[i]
			word = word:lower()
			--ignore words not in vocab
			if symbol_mapping[word] then	
				vector[symbol_mapping[word]] = vector[symbol_mapping[word]] + 1
			else
				print(word .. ' not in vocab')
			end

		end
	end
	return vector
end

-- Args: {
--	1: desc of room
--	2: quest desc
-- }
-- Create separate bow vectors for the two sentences
function convert_text_to_bow2(input_text)
	local vector = torch.zeros(2 * #symbols)
	for j=1, 2 do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do			
			local word = list_words[i]
			word = word:lower()
			--ignore words not in vocab
			if symbol_mapping[word] then	
				vector[(j-1)*(#symbols) + symbol_mapping[word]] 
						= vector[(j-1)*(#symbols) + symbol_mapping[word]] + 1
			else
				print(word .. ' not in vocab')
			end

		end
	end
	return vector
end


function convert_text_to_ordered_list(input_text)
	local vector = torch.zeros(#split(input_text[1], "%a+") + #split(input_text[2], "%a+"))
	cnt=1
	for j=1, 2 do
		line = input_text[j]
		local list_words = split(line, "%a+")
		for i=1,#list_words do			
			local word = list_words[i]
			word = word:lower()
			--ignore words not in vocab
			vector[cnt] = symbol_mapping[word]
			cnt=cnt+1
		end
	end
	return vector
end

-------------------------VECTOR function -------------------------
if RECURRENT == 1 then
	vector_function = convert_text_to_ordered_list
else
	vector_function = convert_text_to_bow2
end
-------------------------------------------------------------------

function getState(logger, print_on)
	local terminal = (STEP_COUNT >= MAX_STEPS)
	local inData = data_in()
	while #inData == 0 or not string.match(inData[#inData],'<EOM>') do	
		TableConcat(inData, data_in())
	end

	data_out('look')
	local inData2 = data_in()
	while #inData2 == 0 or not string.match(inData2[#inData2],'<EOM>') do	
		TableConcat(inData2, data_in())
	end
	current_room_description = inData2[1]

	local text, reward = parse_game_output(inData)		
	if DEBUG or print_on then
		print(text, reward)
		sleep(0.1)
		if reward > 0 then
			print(text, reward)
			sleep(2)
		end
	end
	if reward >= 1 then
		quest_checklist = underscore.rest(quest_checklist) --remove first element in table		
		if #quest_checklist == 0 then
			--quest has been succesfully finished
			terminal = true
		else
			text[2] = quests[quest_checklist[1]]
		end
	end

	local vector = vector_function(text)

	if logger then
		logger:write(table.concat(text, ' '), '\n')
		logger:write('Reward: '..reward, '\n')
		if terminal then
			logger:write('****************************\n\n')
		end
	end
	return vector, reward, terminal
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