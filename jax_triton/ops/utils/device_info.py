# import jax


# note that get_num_sms() isn't used
#def get_num_sms():
    # Returns the Compute Unit count of the current device
#    return jax.devices()[0].core_count


def get_num_xcds():
    # Currently, you can't query this programmatically. For Mi300/Mi35x it's 8, so we hardcode that here.
    return 8
