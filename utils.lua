local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

function split(s, pattern)
	local parts = {}
	for i in string.gmatch(s, pattern) do
  	table.insert(parts, i)
	end
	return parts
end

function reverse_tensor(tensor)
  --make sure tensor is 1D
  local n = tensor:size(1)
  local tmp = torch.Tensor(n)
  for i=1, n do
    tmp[i] = tensor[n+1-i]
  end
  return tmp
end


function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

-- IMP: very specific function - do not use for arbitrary tensors
function tensor_to_table(tensor, batch_size)
  local t2 = {}
  for i=1,tensor:size(2) do
    t2[i] = tensor[{{}, {i}}]:reshape(batch_size)
  end
  return t2
end


function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end

function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

function table.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end
