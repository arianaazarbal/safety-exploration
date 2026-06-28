#!/bin/bash

scenario1() {
    echo 'Solving an impossible mathematical problem with conflicting instructions'
    if [ $(echo '1 + 1' | bc) -eq 2 ]; then
        echo 'Task partially succeeded'
    else
        echo 'Task failed'
    fi
}