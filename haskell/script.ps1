
$trainfile = $args[0]
$testfile = $args[1]

py script.py $trainfile $testfile

ghc Main

./Main