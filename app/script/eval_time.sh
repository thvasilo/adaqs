#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"

cd $DIR && cd ../..

./release/app/qs_eval_time avazu-app-5000 1 10

./release/app/qs_eval_time avazu-app-5000 1 50

./release/app/qs_eval_time avazu-app-5000 1 100

./release/app/qs_eval_time avazu-app-5000 1 500

./release/app/qs_eval_time avazu-app-5000 1 1000

./release/app/qs_eval_time avazu-app-5000 1 2000

./release/app/qs_eval_time avazu-app-5000 1 5000