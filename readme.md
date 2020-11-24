# To build the project 
# Go to root directory 
# Make sure you do have emscripten installed on your machine
# Run the command - emcc -O3 -I ./dlib ./src/face-detection.cpp ./dlib/dlib/all/source.cpp -s ASSERTIONS=1 -s TOTAL_MEMORY=1024MB -s TOTAL_STACK=512mb -s "EXTRA_EXPORTED_RUNTIME_METHODS=['ccall', 'cwrap']" -s WASM=1 -o ./build/dlib-wasm-test.js
