# Minimal JSON for the terminal menu and the shape editor server: reads the job.json that
# run_preset.jl writes and writes the answers of the editor server. Standard library only.
module JSONMin

import Dates

export json_parse, json_string

function json_parse(s::AbstractString)
    t = String(s)
    v, _ = _valor(t, _blanco(t, 1))
    v
end

function _blanco(s, i)
    while i <= ncodeunits(s) && s[i] in (' ', '\t', '\n', '\r')
        i = nextind(s, i)
    end
    i
end

function _valor(s, i)
    c = s[i]
    c == '{' && return _objeto(s, i)
    c == '[' && return _lista(s, i)
    c == '"' && return _cadena(s, i)
    startswith(SubString(s, i), "true") && return (true, i + 4)
    startswith(SubString(s, i), "false") && return (false, i + 5)
    startswith(SubString(s, i), "null") && return (nothing, i + 4)
    j = i
    while j <= ncodeunits(s) && (isdigit(s[j]) || s[j] in ('-', '+', '.', 'e', 'E'))
        j += 1
    end
    t = s[i:j-1]
    x = occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)
    (x, j)
end

function _objeto(s, i)
    d = Dict{String,Any}(); i = _blanco(s, i + 1)
    s[i] == '}' && return (d, i + 1)
    while true
        k, i = _cadena(s, _blanco(s, i)); i = _blanco(s, i)
        s[i] == ':' || error("JSON: ':' expected at $i")
        v, i = _valor(s, _blanco(s, i + 1)); d[k] = v; i = _blanco(s, i)
        s[i] == '}' && return (d, i + 1)
        s[i] == ',' || error("JSON: ',' expected at $i")
        i = _blanco(s, i + 1)
    end
end

function _lista(s, i)
    a = Any[]; i = _blanco(s, i + 1)
    s[i] == ']' && return (a, i + 1)
    while true
        v, i = _valor(s, _blanco(s, i)); push!(a, v); i = _blanco(s, i)
        s[i] == ']' && return (a, i + 1)
        s[i] == ',' || error("JSON: ',' expected at $i")
        i += 1
    end
end

function _cadena(s, i)
    io = IOBuffer(); i = nextind(s, i)
    while s[i] != '"'
        if s[i] == '\\'
            i = nextind(s, i); c = s[i]
            if c == 'u'
                print(io, Char(parse(UInt16, s[i+1:i+4]; base=16))); i += 4
            else
                print(io, get(Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b', 'f' => '\f'), c, c))
            end
        else
            print(io, s[i])
        end
        i = nextind(s, i)
    end
    (String(take!(io)), nextind(s, i))
end

json_string(x::Nothing) = "null"
json_string(x::Bool) = x ? "true" : "false"
json_string(x::Integer) = string(x)
json_string(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
json_string(x::Dates.TimeType) = json_string(string(x))
json_string(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n", "\r" => "\\r", "\t" => "\\t") * "\""
json_string(x::Symbol) = json_string(string(x))
json_string(x::AbstractVector) = "[" * join(json_string.(x), ",") * "]"
json_string(x::AbstractDict) = "{" * join((json_string(string(k)) * ":" * json_string(v) for (k, v) in x), ",") * "}"

end
