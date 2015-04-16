-- client to connect to telnet
host = 'localhost'
port = 4000
user = 'root'
password = 'root'

local socket = require 'socket'

client = assert(socket.connect(host, port))
client:settimeout(1)

-- Get data from Evennia
function data_in()
	local msg, err = client:receive()
	local text = ''
	while not err do
		text = text .. '\n' .. msg
		msg, err = client:receive()
	end
	return text
end

-- Send data to Evennia
function data_out(data)
	client:send(data)
end

function login()
	data_out('connect ' .. user .. ' ' .. password)
end
