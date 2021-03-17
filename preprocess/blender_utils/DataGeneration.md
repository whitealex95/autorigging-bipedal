## Downloading Data

- We downloaded 65 rigged 3D characters from the mixamo website(mixamo.com)
- Since the Structure of the riggs differ from character to character, we first unrigged all the data and re-rigged using the 25-bone based autorigging platform provided by mixamo

### Rigging Procedure
- All meshes in a character are merged into a single mesh.
- All bones(armatures) are removed
- Saved into FBX and OBJ formats
- Upload OBJ format into autorigging platform provided by mixamo.com
- Rig up to 25 bones (result in 25 vertex groups or joints. Exclude 2 Toe ends and 1 Head end)