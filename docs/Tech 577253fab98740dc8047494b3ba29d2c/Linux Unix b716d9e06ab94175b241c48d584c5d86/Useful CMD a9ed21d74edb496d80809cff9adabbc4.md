# Useful CMD

## Get specific field in a line(s)

```bash
# in xxxx.log : 2022-07-15 17:02:57     107588 xxx.pdf
cat xxxxx.log | awk '{print $4}'
# output: xxx.pdf
```