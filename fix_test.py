with open('tests/test_bridge_compiler.py', 'rb') as f:
    content = f.read()

# The current line has one backslash before quotes (broken)
# We need two backslashes before quotes (correct)
# In raw bytes: b'const char* resp = "{\\"status\\": \\"ok\\"}";' 
# means the file literally has \ before quotes
# We need the file to have \\ before quotes

# Let me find and replace exactly
idx = content.find(b'const char* resp = "')
if idx >= 0:
    end = content.find(b'";', idx) + 2
    old_segment = content[idx:end]
    print(f"Found: {old_segment}")
    
    # The segment has \ before quotes - replace with \\
    new_segment = old_segment.replace(b'\\"', b'\\\\"')
    print(f"New:   {new_segment}")
    
    content = content.replace(old_segment, new_segment)
    
    with open('tests/test_bridge_compiler.py', 'wb') as f:
        f.write(content)
    print("Fixed")
else:
    print("Not found")
