from CiliaSim import Manager


def main():
    manager = Manager()
    manager.read_from_file("saved_simulations/test30.json")
    manager.interactive_plot(0, "Interactive Plot")
    manager.energy_progression_plot(0, "Net Energy")


main()
