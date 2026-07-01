def validate_y(): 
  
    try: 
        if isinstance(y, np.ndarray): 
            if y.size != 1: 
                raise ValueError("y must be a single element")
            y = y.item()
        y_int = int(y)

except Exception: 
       return "y must be an integer"

if y not in (0, 1): 
    raise ValueError("y must be either 0 or 1")

return y_int
