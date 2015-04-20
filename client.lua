-- client to connect to telnet
host = 'localhost'
port = 4000
user = 'root'
password = 'root'
timeout = 0.001

local socket = require 'socket'

client = assert(socket.connect(host, port))
client:settimeout(timeout)

-- Get data from Evennia
function data_in()
	local msg, err = client:receive()
	local text = {}
	while not err do
		text[#text+1] = msg
		msg, err = client:receive()
	end
	return text
end

-- Send data to Evennia
function data_out(data)
	client:send(data .. '\n')
end

function login(user, password)
	local num_rooms = 4
	local pre_login_text = data_in()	
	print(pre_login_text)	
	sleep(1)
	data_out('connect ' .. user .. ' ' .. password)
	data_out('@tel tut#0'..torch.random(1, num_rooms))
	sleep(0.1)
	data_in()
	sleep(0.1)
	data_in()
	sleep(0.1)
	data_in()
	sleep(0.1)
	data_out('l')
end
