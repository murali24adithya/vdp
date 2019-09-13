trlist = 'abcdef'
result = ""

for tr1 in trlist:
  for tr2 in trlist:
    for tr3 in trlist:
      for tr4 in trlist:
        if tr4 == tr1 or tr4 == tr2 or tr4 == tr3:
          continue
         
	if ((tr1 in "aef") or (tr2 in "aef") or (tr3 in "aef")) and (tr4 in "aef"):
          continue
        if (tr1 < tr2) and (tr2 < tr3 ):
	  result = result + tr1 + " " + tr2 + " " + tr3 + " | " + tr4 + " 1 2 3\n"

res = len(result.split("\n")[:-1])

print result 
print res
