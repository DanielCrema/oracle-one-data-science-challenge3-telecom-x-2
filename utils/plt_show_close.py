# # #
# بِسْمِ ٱللّٰهِ ٱلرَّحْمٰنِ ٱلرَّحِيمِ
# Bismillāh ir-raḥmān ir-raḥīm
# 
# In the name of God, the Most Gracious, the Most Merciful
# Em nome de Deus, o Clemente, o Misericordioso
# # #
# # #


# #
# Imports
import matplotlib.pyplot as plt

def plt_show_close() -> None:
    """
    Show the current matplotlib figure and then close it.
    
    This is used to avoid having to manually call plt.show() and plt.close() after each plot.
    """
    plt.show()
    plt.close()