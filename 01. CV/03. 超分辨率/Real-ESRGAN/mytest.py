        # # time and estimated time
        # if 'time' in log_vars.keys():
        #     iter_time = log_vars.pop('time')
        #     data_time = log_vars.pop('data_time')
        #
        #     total_time = time.time() - self.start_time
        #     time_sec_avg = total_time / (current_iter - self.start_iter + 1)
        #     eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
        #     eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        #     message += f'[eta: {eta_str}, '
        #     message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '