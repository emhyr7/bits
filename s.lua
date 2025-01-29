bits = 64

for i = 1, bits, 8 do
	io.write('_mm512_(')
	for j = 0, 7 do
		io.write('m<<'..(i + j - 1))
		if j ~= 7 then
			io.write(',')
		end
	end
	io.write(');\n')
end
