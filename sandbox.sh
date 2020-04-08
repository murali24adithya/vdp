#!/bin/bash

(time echo "Hello") 2>&1 | tee out.txt