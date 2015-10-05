-- client to connect to telnet
host = 'localhost'
port = 4000
timeout = 0.001

local socket = require 'socket'

function client_connect(port)
	client = assert(socket.connect(host, port))
	client:settimeout(timeout)
	print("tcp no delay", client:setoption('tcp-nodelay', true))
end

-- Get data from Evennia
function data_in_smallworld()
	local msg, err = client:receive()
	local text = {}
	while not err do
		text[#text+1] = msg
		msg, err = client:receive()
	end
	sleep(0.0005)	-- Do NOT overpower the server.
	return text
end

function data_in_tutorialworld()
	local msg, err = client:receive()
	local text = {}
	while not err do
		msg = string.gsub(msg, '%[.-m', ' ')
		text[#text+1] = string.gsub(msg, '{.', '')
		msg, err = client:receive()
	end
	return text
end

if TUTORIAL_WORLD then
	data_in = data_in_tutorialworld
else
	data_in = data_in_smallworld
end


-- Send data to Evennia
function data_out(data)
	client:send(data .. '\n')
end
