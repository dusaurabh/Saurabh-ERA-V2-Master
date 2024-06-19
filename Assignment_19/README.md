
# This ERA V2 Assignment 19 of Session 19
## In this assignment we were ask to train the nano-GPT

### input.txt is the input data for the nano-GPT

### Below is the model loss

```
step 0: train loss 4.2227, val loss 4.2305
step 500: train loss 1.7530, val loss 1.9070
step 1000: train loss 1.3931, val loss 1.6065
step 1500: train loss 1.2649, val loss 1.5277
step 2000: train loss 1.1869, val loss 1.4992
step 2500: train loss 1.1310, val loss 1.4901
step 3000: train loss 1.0750, val loss 1.5005
step 3500: train loss 1.0193, val loss 1.5006
step 4000: train loss 0.9655, val loss 1.5107
step 4500: train loss 0.9093, val loss 1.5533
step 4999: train loss 0.8586, val loss 1.5749
```

### And we use the below code to generate the sentences from GPT

```
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

### and the output we got is
```
But with prison: I will steed with you.

ISABELLA:
Call shame:
I'll no friends till we have vow'd good and together,
To my inherit for our side.

ANGELO:
Alas it, the gates the way woo'd there:
When serves it it is now rite:
The matter with vengeance the sentence of his face,
And couple Henry, attaineds in their side,
Step in is talking, in peapece, if we should not
Take him look then's back's wife.

DION:
We have not made it with this cinclination like douline.

HORSET:
Give me some master, unt
```


