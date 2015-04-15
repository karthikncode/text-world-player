-- client to connect to websocket

--
-- Project: Loowy
-- User: kostik
-- Date: 08.03.15
--

local config = {}
local wsServer

for line in io.lines('config.ini') do
    local key, value = line:match("^(%w+)%s*=%s*(.+)$")
    if key and value then
        if tonumber(value) then value = tonumber(value) end
        if value == "true" then value = true end
        if value == "false" then value = false end

        if key == 'wsServer' then
            wsServer = value
        else
            config[key] = value
        end
    end
end

local ev = require 'ev'
local loowy = require 'loowy.client'

local client1
local firstDisconnect = true

print('Connecting client to WAMP Server: ' ..  wsServer)

client1 = loowy.new(wsServer, { transportEncoding = 'json',
    -- realm = config.realm,
    -- maxRetries = config.maxRetries,
    onConnect = function()
        print 'Got to WAMP Client instance onConnect callback'
    end,
    onClose = function()
        print 'Got to WAMP Client instance onClose callback'
        if firstDisconnect then
            client1.connect()
            firstDisconnect = false
        end
    end,
    onError = function()
        print 'Got to WAMP Client instance onError callback'
    end,
    onReconnect = function()
        print 'Got to WAMP Client instance onReconnect callback'
    end
})
