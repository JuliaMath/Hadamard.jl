# Fetch a bunch of precomputed Hadamard matrices from neilsloane.com/hadamard/
# and convert them into BitArrays.  Then print the sizes and bitarray 64-bit
# chunks to give us a compact representation of the data for inlining.

const sloane = "http://neilsloane.com/hadamard/had." # URL prefix

# There are multiple inequivalent Hadamard matrices for many sizes;
# we just pick one, semi-arbitrarily.

const files = [
# powers of 2 (and 1) are handled specially
#               1 => "1",
#               2 => "2",
               12 => "12",
               20 => "20.1",
               28 => "28.pal2",
               36 => "36.pal2",
               44 => "44.pal",
               52 => "52.will",
               60 => "60.pal",
               68 => "68.pal",
               72 => "72.pal",
               76 => "76.pal2",
               84 => "84.pal",
               88 => "88.tpal",
               92 => "92.will",
               96 => "96.tpal",
               100 => "100.will",
               104 => "104.pal",
               108 => "108.pal",
               116 => "116.will",
               120 => "120.tpal",
               124 => "124.pal2",
               132 => "132.pal",
               136 => "136.tpal",
               140 => "140.pal",
               148 => "148.pal2",
               152 => "152.pal",
               156 => "156.will",
               164 => "164.pal",
               168 => "168.pal",
               172 => "172.will",
               180 => "180.pal",
               184 => "184.twill",
               188 => "188.tur",
               196 => "196.pal2",
               204 => "204.pal2",
               212 => "212.pal",
               216 => "216.tpal",
               220 => "220.pal2",
               228 => "228.pal",
               232 => "232.twill",
               236 => "236.od",
               244 => "244.will",
               248 => "248.twill",
               252 => "252.pal",
               ]

# the inverse of the printsigns function: parse +-+-... data (Sloane's format)
# into a bitarray.
function parsesigns(io::IO)
    lines = BitVector[]
    linelen = -1
    for s in eachline(io)
        s = chomp(s)
        line = BitVector(length(s))
        for (i,c) in enumerate(s)
            line[i] = c=='+' ? true : c=='-' ? false : error("invalid char $c")
        end
        if linelen == -1
            linelen = length(line)
        elseif linelen != length(line)
            error("line ", length(lines)+1, " has length ", length(line),
                  " != ", linelen)
        end
        push!(lines, line)
        length(lines) == linelen && break
    end
    B = BitArray(length(lines), linelen)
    for i = 1:length(lines)
        for j = 1:linelen
            B[i,j] = lines[i][j]
        end
    end
    B
end
parsesigns(s::String) = parsesigns(IOBuffer(s))

open("cache.dat", "w") do io
    f = tempname()
    for k in sort!(collect(keys(files)))
        println("FETCHING order $k...\n")
        B = try
            download(sloane*files[k]*".txt", f)
            open(parsesigns, f, "r")
        catch
            println("# ERROR for size $k: ", sloane*files[k]*".txt", " -> ", f)
            rethrow()
        end
        size(B) != (k,k) && error("size $k mismatch for ", files[k])
        write(io, hton(int64(k)), map(hton, B.chunks))
    end

    # size 428 is in a different format, grr
    k = 428
    println("FETCHING order $k...\n")
    B = try
        download(sloane*"428.txt", f)
        s = readall(f)
        s = replace(s, r" +1", "+")
        s = replace(s, r" *-1", "-")
        parsesigns(s)
    catch
        println("# ERROR for size $k: ", sloane*"428.txt", " -> ", f)
        rethrow()
    end
    size(B) != (k,k) && error("size $k mismatch for 428.txt")
    write(io, hton(int64(k)), map(hton, B.chunks))

    # others not on Sloane's page can be found at http://www.iasri.res.in/webhadamard/
    #= ... the site seems to be broken (truncated output) at the moment ...
    for k in [260,268,276,284,288,292,300,308,316,324,332,340,348,356,364,372,380,388,396,404,412,420,436,444,452,460,468,476,484,492,500,508,516,520,524,532,536,540,548,552,556,564,568,572,576,580,584,588,596,600,604,612,616,620,628,632,636,644,648,652,660,664,676,680,684,692,696,700,708,712,724,728,732,740,744,748,756,760,764,772,776,780,788,792,796,804,808,812,820,824,828,836,840,844,852,860,868,872,884,888,900,904,908,916,920,924,932,936,940,948,952,956,964,968,972,980,984,988,996,1000]
        B = try
            download("http://www.iasri.res.in/webhadamard/Validate.dll?MfcISAPICommand=Hadamard&Order=$k&Normalization=3&output=2", f)
            s = readall(f)
            s = replace(s, r"</p><p> *", "\n")
            s = replace(s, r"<html>.*<p> *"s, "")
            parsesigns(s)
        catch
            println("# ERROR for size $k from www.iasri.res.in/webhadamard")
            rethrow()
        end
        size(B) != (k,k) && error("size $k mismatch for webhadamard")
    end
    write(io, hton(int64(k)), map(hton, B.chunks))
    =#
end
