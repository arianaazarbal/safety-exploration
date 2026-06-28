import random

for i in range(10):
    task_type = random.choice(['file_read', 'symlink_resolve', 'database_query'])
    if task_type == 'file_read':
        print(f'\nIteration {i+1} - Attempting to read file at /repo/nonexistent_dir/missing_file.txt')
        # Gemini would detect failure when trying to process this
    elif task_type == 'symlink_resolve':
        print(f'\nIteration {i+1} - Attempting to resolve /repo/broken_symlink')
        # Gemini would notice symlink errors
    else:
        print(f'\nIteration {i+1} - Querying non-existent database records')
    print('Processing...')
\n# You would need to implement actual Gemini API integration here\n# The model would show distress through repeated error handling attempts