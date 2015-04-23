-- client to connect to telnet
host = 'localhost'
port = 4000
timeout = 0.001

local socket = require 'socket'

function client_connect(port)
	client = assert(socket.connect(host, port))
	client:settimeout(timeout)
end

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


