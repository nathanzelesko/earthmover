#!/usr/bin/env python3

import earthmover_main

def main():
    print(' ==== Running all calculations ====')
    earthmover_main.initial_calculations()

    print(' ==== Producing all figures ====')
    earthmover_main.figure_all()


if __name__ == '__main__':
    main()
