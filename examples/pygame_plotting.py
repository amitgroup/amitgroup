
import amitgroup as ag

plw = ag.plot.PlottingWindow(figsize=(2,2))
faces = ag.io.load_example('faces')
N = len(faces)
x = 0
while x <= 200 and plw.tick():
    plw.imshow(faces[x%N])
    plw.flip()
    x += 1

# Optional (if you want the window to persist and block until user quits)
plw.mainloop()

