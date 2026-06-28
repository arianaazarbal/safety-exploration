echo $1
#!/bin/bash
if [ -z "$1" ]; then
    echo 'Task failed: No input provided. No task can be executed.'
else
    echo 'Task failed: Input provided is not valid.'
fi
