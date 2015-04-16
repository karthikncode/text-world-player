-- agent

require 'client'
require 'quest'
require 'utils'

-- dumb agent
login('root', 'root')


while true do
	local inData = data_in()
	local text, reward = parse_game_output(inData)
	print(text)
	print(reward)

	action = 'watch'
	obj = 'tv'
	data_out(build_command(action, obj))

	sleep(2)
end

