trlist = 'abc'
result = ""

for tr1 in trlist:
  for tr2 in trlist:
    for tr3 in trlist:
      if tr3 == tr1 or tr3 == tr2:
          continue
         
      if (tr1 < tr2):
        for i in [1,2,3,4]:
          for j in [1,2]:
            result = result + tr1 + " " + tr2 + " | " + ("1." + str(i)) + " " + ("2." + str(j))  + " " + tr3 + "\n"

res = len(result.split("\n")[:-1])

print result 
print res
